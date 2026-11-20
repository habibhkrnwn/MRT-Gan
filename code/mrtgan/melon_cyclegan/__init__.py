"""CycleGAN maturity-translation experiment for the melon dataset.

The package implements the protocol in `RENCANA_CYCLEGAN.md`: ROI crops of the
frozen train split feed an unpaired ripe/unripe translation model, the accepted
translations become detector arms, and the arms are compared against the arms
that already exist in `04_Hasil/detector_ablation_v2/` under the locked detector
protocol.

Domain naming is fixed everywhere in this package:

    domain A = unripe = class_id 1
    domain B = ripe   = class_id 0
    G_A : A -> B = unripe -> ripe
    G_B : B -> A = ripe -> unripe

Training code for the networks themselves is the vendored upstream repository
under `vendor/pytorch_cyclegan`; this package only adds the dataset, the
generator variant, the extra losses and the downstream tooling.
"""

__all__ = ["config", "color", "crops", "losses", "networks", "dataset_impl", "model_impl",
           "generate", "qc", "train_loop"]
