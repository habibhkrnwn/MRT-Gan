"""Training-loss curves (seed 11, logged every 100 iterations) for MRT-GAN, its texture ablation and the initial
configuration -> figures/data/generator_loss_r512.dat. Columns follow figures/data/generator_loss.dat:
G_B = adversarial term of the ripe-to-unripe generator, D_A = loss of the global unripe discriminator,
cycle_B = ripe-domain cycle term, texture_* = local texture terms (absent without texture discriminators: nan)."""
import re
from pathlib import Path
HERE = Path(__file__).resolve().parent; M = HERE.parents[1]; G = M.parents[1]
LOGS = {'MRT': G / 'runs/16_g3ti_r512/kaggle_v1_output/g3ti_r512_v1/result/train_G3TI_R512.log',
        'NOT': G / 'runs/16_g3ti_r512/kaggle_v1_output/g3ti_r512_v1/result/train_G3TI_noT_R512.log',
        'INIT': G / 'runs/05_texture_study/kaggle_v3_output/texture_study_v1/study/p100/logs/G3.log'}
KEYS = ['G_B', 'D_A', 'cycle_B', 'idt_B', 'texture_G_B', 'texture_D_A']
data = {}
for tag, path in LOGS.items():
    for line in path.read_text().splitlines():
        m = re.match(r'epoch (\d+) iter (\d+) ', line)
        if m:
            vals = dict((k, float(v)) for k, v in re.findall(r'(\w+):([-\d.]+)', line))
            data.setdefault(int(m.group(2)), {})[tag] = vals
iters = sorted(i for i, d in data.items() if all(t in d for t in LOGS))
cols = [f'{k}_{t}' for k in KEYS for t in LOGS]
rows = [' '.join(['iters'] + cols)]
for i in iters:
    rows.append(' '.join([str(i)] + [f"{data[i][t].get(k, float('nan')):.6f}" for k in KEYS for t in LOGS]))
(M / 'figures/data/generator_loss_r512.dat').write_text('\n'.join(rows) + '\n')
print(len(iters), 'rows; first iters', iters[:3], 'last', iters[-1])
