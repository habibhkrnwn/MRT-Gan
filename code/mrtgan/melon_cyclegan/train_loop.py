"""Training entry point: the upstream loop without the visualisation stack.

`vendor/pytorch_cyclegan/train.py` imports visdom, dominate and wandb at module
level. None of them is needed to train and none is guaranteed on Kaggle, so this
module runs the same loop and writes the losses to `loss.json` instead. Models,
datasets, options, schedulers and checkpoint handling all come from upstream
unchanged.

    python -m melon_cyclegan.train_loop --dataroot ... --name ... --model ...
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .config import ROOT, vendor_on_path

vendor_on_path()


def _pop(flag, cast=float, default=None):
    """Read and remove one flag upstream does not know about."""
    if flag not in sys.argv:
        return default
    index = sys.argv.index(flag)
    value = cast(sys.argv[index + 1])
    del sys.argv[index:index + 2]
    return value


def seed_everything(seed):
    """Seed python, numpy and torch, including dataloader workers.

    PyTorch seeds each worker's `random` module from the base generator, so the
    unaligned pairing and the horizontal flips become reproducible once the main
    process is seeded.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device():
    import torch

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_samples(model, out, epoch):
    """Write the current visuals as PNG so training can be inspected afterwards."""
    import numpy as np
    from PIL import Image

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, tensor in model.get_current_visuals().items():
        array = tensor[0].detach().float().cpu().clamp(-1, 1).numpy()
        array = ((array + 1.0) * 127.5).round().astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(array).save(out / f"epoch{epoch:04d}_{name}.png")


def monitor_panel(dataroot, count, seed):
    """Fixed TRAIN inputs, independent of loader sampling; no model selection."""
    import random
    from .dataset_impl import MelonRoiData, load_pair

    data = MelonRoiData(dataroot, no_flip=True)
    rng = random.Random(seed)
    paths = [rng.sample(files, min(count, len(files))) for files in (data.A_paths, data.B_paths)]
    panel = []
    for a, b in zip(*paths):
        x, mx = load_pair(a, data.mask_for(a, 'A'))
        y, my = load_pair(b, data.mask_for(b, 'B'))
        panel.append(dict(A=x.unsqueeze(0), B=y.unsqueeze(0), A_mask=mx.unsqueeze(0), B_mask=my.unsqueeze(0),
                          A_paths=[str(a)], B_paths=[str(b)]))
    return panel


def save_monitor_panel(model, panel, out, epoch):
    import torch
    import numpy as np
    from PIL import Image

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    networks = [getattr(model, 'net' + name) for name in model.model_names]
    modes = [net.training for net in networks]
    try:
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(panel):
                model.set_input(sample)
                model.forward()
                for name in ('real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B'):
                    x = getattr(model, name)[0].detach().cpu().clamp(-1, 1)
                    image = ((x.numpy() + 1) * 127.5).round().astype(np.uint8).transpose(1, 2, 0)
                    Image.fromarray(image).save(out / f'epoch{epoch:04d}_sample{i:02d}_{name}.png')
    finally:
        for net, training in zip(networks, modes):
            net.train(training)


def main(argv=None):
    if argv is not None:
        sys.argv = [sys.argv[0]] + list(argv)
    from data import create_dataset
    from models import create_model
    from options.train_options import TrainOptions

    budget_minutes = _pop("--time_budget_minutes")
    seed = _pop("--seed", int)
    sample_freq = _pop("--sample_epoch_freq", int, 0)
    monitor_count = _pop('--monitor_samples', int, 0)
    if seed is not None:
        seed_everything(seed)
    opt = TrainOptions().parse()
    opt.device = _device()
    dataset = create_dataset(opt)
    size = len(dataset)
    print(f"The number of training images = {size}", flush=True)

    model = create_model(opt)
    model.setup(opt)
    if getattr(opt, 'controlled_init_seed', -1) >= 0:
        # Model construction must not alter the data/augmentation stream between arms.
        seed_everything(seed if seed is not None else opt.controlled_init_seed)
        import hashlib
        initial = {}
        for name in ('G_A', 'G_B', 'D_A', 'D_B'):
            net = getattr(model, 'net' + name)
            h = hashlib.sha256()
            for key, value in net.state_dict().items():
                h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
            initial[name] = h.hexdigest()
        (Path(opt.checkpoints_dir) / opt.name / 'initial_weights.json').write_text(
            json.dumps(initial, indent=2))
    panel = monitor_panel(opt.dataroot, monitor_count, seed or 0) if monitor_count else []
    if panel:
        panel_path = Path(opt.checkpoints_dir) / opt.name / 'monitor_panel.json'
        panel_path.write_text(json.dumps([dict(A=Path(s['A_paths'][0]).stem, B=Path(s['B_paths'][0]).stem)
                                         for s in panel], indent=2) + '\n')

    history, total_iters, started = [], 0, time.time()
    last_epoch = opt.n_epochs + opt.n_epochs_decay
    deadline = started + budget_minutes * 60 if budget_minutes else None
    stopped = None
    for epoch in range(opt.epoch_count, last_epoch + 1):
        epoch_started = time.time()
        for data in dataset:
            total_iters += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            if total_iters % opt.print_freq == 0:
                losses = {k: float(v) for k, v in model.get_current_losses().items()}
                history.append(dict(epoch=epoch, iters=total_iters, **losses))
                report = " ".join(f"{k}:{v:.3f}" for k, v in losses.items())
                print(f"epoch {epoch} iter {total_iters} {report}", flush=True)
        if sample_freq and (epoch % sample_freq == 0 or epoch == last_epoch):
            if panel:
                save_monitor_panel(model, panel, Path(opt.checkpoints_dir) / opt.name / 'samples', epoch)
            else:
                save_samples(model, Path(opt.checkpoints_dir) / opt.name / "samples", epoch)
        if epoch % opt.save_epoch_freq == 0 or epoch == last_epoch:
            model.save_networks("latest")
            model.save_networks(epoch)
        model.update_learning_rate()
        print(f"end of epoch {epoch}/{last_epoch} in {time.time() - epoch_started:.1f}s", flush=True)
        if deadline and time.time() > deadline:
            # A run that runs out of wall clock saves what it has and says so; the
            # shortened schedule is recorded, never presented as the planned one.
            stopped = dict(reason="time budget reached", epoch=epoch, planned_epochs=last_epoch)
            model.save_networks("latest")
            print(f"stopping early at epoch {epoch}/{last_epoch}: time budget reached", flush=True)
            break

    out = Path(opt.checkpoints_dir) / opt.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "loss.json").write_text(json.dumps(dict(
        name=opt.name, model=opt.model, netG=opt.netG, seed=seed,
        epochs=last_epoch, dataset_size=size, stopped=stopped,
        seconds=round(time.time() - started, 1), history=history), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
