"""Region-aligned cross attention, reimplemented from the ReCon paper.

Clean-room: written from the published method (NeurIPS 2025, arXiv 2510.15783).
No upstream source is vendored; see provenance/recon_clean_room.json.

Each spatial query token attends only to the text embedding of the region it
falls in; tokens outside every region attend to a single context embedding.

The published implementation smuggles the region masks and the per-region
embedding stack through ``encoder_hidden_states`` as a dict, which requires
forking UNet2DConditionModel, Transformer2DModel, the UNet blocks and
ControlNetModel so a non-tensor can survive the journey. That fork cannot be
reused here, so instead the regions are attached to the processor itself with
``set_regions`` before sampling and diffusers keeps receiving an ordinary
tensor. Same behaviour, no forked stack.

Upstream selects a region's key with ``i < num_regions - 1``, which routes the
last region of every image to the context slot instead of its own class slot.
With one object per image that is every image. ``per_region`` corrects it,
``upstream`` reproduces it for a fidelity check, and ``disabled`` drops region
routing entirely as the ablation control.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

MODES = ('per_region', 'upstream', 'disabled')


def region_key_index(i, n_region, mode='per_region'):
    """Key slot a region's query tokens attend to. Context lives at index n_region."""
    if mode not in MODES: raise ValueError(f'Unknown raca_indexing: {mode}')
    if mode == 'disabled': raise ValueError('disabled mode does not route regions')
    if mode == 'upstream': return i if i < n_region - 1 else n_region
    return i


def resize_region_masks(region_masks, tokens):
    """Nearest-resize each (H,W) mask to the attention grid, flattened to `tokens`.

    Overlaps resolve first-come: a token claimed by an earlier region is not
    reclaimed, so regions stay mutually exclusive at every resolution.
    """
    side = int(round(tokens ** .5))
    if side * side != tokens: raise ValueError(f'Attention grid is not square: {tokens} tokens')
    if not region_masks: raise ValueError('No region masks supplied')
    device = region_masks[0].device
    flat, claimed = [], torch.zeros(tokens, dtype=torch.bool, device=device)
    for mask in region_masks:
        m = mask.detach().to(torch.float32)
        while m.ndim > 2: m = m.squeeze(0)
        if m.ndim != 2: raise ValueError(f'Region mask must be 2D, got {tuple(mask.shape)}')
        if m.shape != (side, side):
            m = F.interpolate(m[None, None], size=(side, side), mode='nearest')[0, 0]
        keep = (m.reshape(-1) != 0) & (~claimed)
        claimed = claimed | keep
        flat.append(keep)
    return flat, ~claimed


class RegionAlignedAttnProcessor:
    """Diffusers attention processor. Self-attention is untouched.

    Call ``set_regions(masks, embeds)`` before sampling, where ``embeds`` is
    (batch, n_region + 1, tokens, dim): one slot per region then the context
    slot. Cross-attention then routes by region; with no regions set, or in
    ``disabled`` mode, behaviour is ordinary cross-attention on whatever
    diffusers passed in.
    """

    def __init__(self, mode='per_region'):
        if mode not in MODES: raise ValueError(f'Unknown raca_indexing: {mode}')
        self.mode = mode
        self.region_masks = None
        self.region_embeds = None
        self.routed_slots = []  # last call's routing; tests and provenance read it

    def set_regions(self, masks, embeds):
        """Attach per-region masks and their embedding stack for the next sampling pass."""
        if masks is not None and embeds is not None:
            if embeds.ndim != 4:
                raise ValueError(f'Region embeds must be (batch, slots, tokens, dim), got {tuple(embeds.shape)}')
            if embeds.shape[1] != len(masks) + 1:
                raise ValueError(f'Expected {len(masks) + 1} slots for {len(masks)} regions, got {embeds.shape[1]}')
        self.region_masks, self.region_embeds = masks, embeds
        return self

    def clear_regions(self):
        self.region_masks = self.region_embeds = None
        return self

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 temb=None, *args, **kwargs):
        residual = hidden_states
        if getattr(attn, 'spatial_norm', None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        is_cross = encoder_hidden_states is not None
        routed = (is_cross and self.mode != 'disabled'
                  and self.region_masks is not None and self.region_embeds is not None)
        if not is_cross: encoder_hidden_states = hidden_states
        if routed: encoder_hidden_states = self.region_embeds.to(hidden_states.dtype)

        batch, tokens, _ = hidden_states.shape
        if getattr(attn, 'group_norm', None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if is_cross and getattr(attn, 'norm_cross', None) and not routed:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        query = query.view(batch, tokens, attn.heads, head_dim).transpose(1, 2)

        self.routed_slots = []
        if routed:
            n_region = len(self.region_masks)
            n_key = encoder_hidden_states.shape[1]
            key = key.view(batch, n_key, -1, attn.heads, head_dim).transpose(2, 3)
            value = value.view(batch, n_key, -1, attn.heads, head_dim).transpose(2, 3)
            selected, background = resize_region_masks(
                [m.to(query.device) for m in self.region_masks], tokens)
            out = torch.zeros_like(query)
            for i, keep in enumerate(selected):
                if not bool(keep.any()): continue
                slot = region_key_index(i, n_region, self.mode)
                self.routed_slots.append((i, slot, int(keep.sum())))
                out[:, :, keep] = F.scaled_dot_product_attention(
                    query[:, :, keep], key[:, slot], value[:, slot], dropout_p=.0)
            if bool(background.any()):
                self.routed_slots.append(('background', n_region, int(background.sum())))
                out[:, :, background] = F.scaled_dot_product_attention(
                    query[:, :, background], key[:, n_region], value[:, n_region], dropout_p=.0)
            hidden_states = out
        else:
            key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch, tokens, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)
        if getattr(attn, 'residual_connection', False): hidden_states = hidden_states + residual
        return hidden_states / getattr(attn, 'rescale_output_factor', 1.)
