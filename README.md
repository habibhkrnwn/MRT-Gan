# MRT-GAN: Fruit-Texture Discrimination for Melon Ripeness Transformation with Shortcut-Aware Evaluation

Code, notebooks and released evidence for the manuscript

> H. H. Kurniawan, T. A. Sardjono, and H. Widyantara, "MRT-GAN: Fruit-Texture Discrimination for Melon Ripeness
> Transformation with Shortcut-Aware Evaluation", *International Journal of Intelligent Engineering and Systems*
> (manuscript 20267304, under review, revision 1).

MRT-GAN is a CycleGAN that translates ripe golden-melon fruit into unripe fruit to add minority-class images to a
YOLOv8n ripeness detector. Compared with the initial configuration it is trained at 512 pixels, composites the
fruit mask only at export, uses identity weight 0.5, and adds local RGB texture discriminators on fruit-interior
patches. The paper also evaluates whether the detector relies on the image background (shortcut probes,
background-swap stress test, exposure and scene controls, and a blinded farmer assessment).

The dataset is the public Puspalebo golden-melon dataset from its authors (Umar et al.), available at
<https://github.com/ubaidillahumar/Melon-Ripeness-Detection-Dataset>. It is not redistributed here.

## What is where

| Item | Location |
|---|---|
| MRT-GAN package, training/export tools, configs, unit tests | `code/mrtgan/` |
| Detector training, screening and ranking utilities | `code/detector/` |
| Screening-pack builders and screening summaries | `code/screening/` |
| Kaggle notebooks with their `kernel-metadata.json` (GPU runs) | `notebooks/` |
| Local scripts that compute the tables and figures | `analysis/` |
| Old-to-new path map and SHA-256 manifest of copied files | `docs/` |
| Generator and detector weights, all generated candidates, fruit masks, split identifiers | GitHub Release of this repository (see below) |
| Screening records, farmer ratings, per-image predictions | `evidence/` where released; otherwise from the corresponding author on request |

## Repository layout

```
code/
  mrtgan/                 melon_cyclegan package (generator, texture discriminators, losses, export, QC)
    config/               training configurations used by the notebooks
    tools/                command-line tools (crops, training, candidate export, colour margin, arms)
    tests/                unit tests for the texture discriminator and texture study
    vendor/pytorch_cyclegan/  upstream pytorch-CycleGAN-and-pix2pix (BSD licence, kept with its LICENSE)
  detector/               YOLOv8n training, detector arms, screening and ranking (src/, tools/)
  screening/              blinded screening packs (MRT-GAN, colour transfer) and screening summaries
notebooks/
  generator/              13, 22-26 (MRT-GAN chain), 29 (matched 512-px ablations),
                          34 (SD-1.5 inpainting + LoRA), 35 (AnchorDiffusion), 32 (CycleGAN-Turbo, not completed)
  detector/
    S1_main_arms/         27 (four arms x five seeds), 28 (fruit-only detectors)
    S2_k_paste_real/      30 (paste_real control and k = 11, 22, 44)
    S3_inpainting_baseline/   33 (detector arms for the inpainting baseline)
    S4_anchordiffusion_baseline/  36 (baseline comparison), 37 (held-out folds for the baselines)
    heldout_dsc_folds/    31 (within-source held-out DSC capture-group folds)
    yolo_versions/        21 (YOLO generation comparison)
  development/            08-20, earlier development stages kept for provenance
analysis/
  table1_partition/ ... table6_farmer_assessment/   scripts grouped by the table they produce
  figures/, paper_build/, sensitivity_near_duplicates/, revision_checks/, decision_layer/, yolo_versions/
configs/detector/         detector-session configurations (S1-S4)
evidence/                 released evidence files (see "Evidence" below)
docs/
  PATH_MAP_old_to_new.tsv     new path, SHA-256 and the path in the earlier repository
  SOURCE_MANIFEST_old_repo.tsv  SHA-256 and original project location of every copied file
```

## Paper-to-code map

| Paper item | Notebook(s) | Analysis script(s) |
|---|---|---|
| Table 1, partition and capture sessions | – | `code/mrtgan/tools/audit_split_leakage.py`, `analysis/sensitivity_near_duplicates/sens.py` |
| Fig. 2, MRT-GAN architecture | – | `analysis/table2_generator_ablation/generator_r512/build_generator_figures.py` |
| Table 2, generator ablation and baselines | `generator/13`, `22`-`26`, `29`, `34`, `35` | `analysis/table2_generator_ablation/` (FID/KID, spectral and colour diagnostics, `nyquist_seed_ratios.py`) |
| Fig. 3, ablation examples | `generator/25`, `26` | `analysis/paper_build/build_image_figures.py` |
| Table 3, shortcut probes | `detector/S1_main_arms/27` | `analysis/table3_shortcut_probes/` |
| Table 4, background swap and fruit-only | `detector/S1_main_arms/27`, `28_fruitonly_m00`, `28_fruitonly_m15` | `analysis/table4_background_swap/` |
| Fig. 4, background-swap examples | – | `analysis/table4_background_swap/build_swap_figure.py` |
| Table 5, exposure and scene controls | `detector/S2_k_paste_real/30`, `heldout_dsc_folds/31`, `S3_inpainting_baseline/33`, `S4_anchordiffusion_baseline/36`, `37` | `analysis/table4_background_swap/bg_swap.py` applied to each session |
| Table 6, blinded farmer assessment | – | `analysis/table6_farmer_assessment/` |
| Fig. 5, Grad-CAM | `detector/S1_main_arms/27` | `analysis/figures/detector_postanalysis/build_gradcam_r512.py` |
| Screening counts (22/217, 56/217) and agent-log audit | `development/15`, `19` | `code/screening/screening_summary.py`, `code/screening/agent_log_audit.py` |
| Exploratory decision layer (Section 4.8) | – | `analysis/decision_layer/` |

Internal run names: `G3TI_R512` (and `_s12`, `_s13`, `MRTGAN_s12` in detector runs) is MRT-GAN;
`G3TI_noT_R512` is MRT-GAN without the RGB texture discriminators; `G3` is the initial configuration;
`CT`/`colour_transfer` is the colour-transfer baseline; `INPAINT` and `ANCHOR` are the diffusion baselines.
Detector arms in kernel 27: `real_only`, `oversample_k22`, `colour_transfer_k22`, `mrtgan_k22`; detector seeds 0-4.

## Released data and models

The GitHub Release of this repository holds the files that are too large for git:

| File | Contents |
|---|---|
| `weights.zip` | MRT-GAN EMA generator weights, seeds 11, 12, 13, and the generators without RGB texture discriminators |
| `detectors_kernel27_selected.zip` | Selected YOLOv8n weights per arm and seed (session S1) |
| `candidates.zip` | All 217 generated candidates per generator and ablation (Table 2) |
| `candidates_inpaint_baseline.zip`, `candidates_anchor_baseline.zip` | Candidates of the two diffusion baselines (Table 2) |
| `masks_and_splits.zip` | Training fruit masks, crop manifest, colour statistics, photograph split, capture-group and near-duplicate lists |
| `*_result.zip` (kernels 29, 30, 31, 33, 36, 37) | Result archives of the revision runs, with SHA-256 files |
| `SHA256SUMS` | Checksums; verify with `sha256sum -c SHA256SUMS` |

The released images and masks are derived from the Puspalebo dataset and keep its CC BY 4.0 licence.

## Evidence

`evidence/` holds the small files behind the reported numbers: split and exclusion lists, detector seed and
checkpoint mappings, calibration and threshold records, lock files, and per-image predictions for the detector
sessions. Screening records, human ratings and any file that would unblind the assessors are available to the
editor and reviewers from the corresponding author on request.

## Environment

- GPU runs: Kaggle notebooks (2 x Tesla T4) on the Kaggle Python image; the image version is recorded in each
  kernel log. Detector runs pin `ultralytics==8.4.163`.
- Local analysis: Python 3.12, CPU. Install with `pip install -r requirements.txt` (pinned versions).
- FID and KID: `torch-fidelity 0.4.0`, KID subsets of 80 images and 100 subsets.

## Reproducing the results

1. Download the Puspalebo dataset and the release files listed above.
2. Upload `code/` as a Kaggle dataset (the notebooks attach it as `melon-texture-study-code`) and run the notebooks
   in numerical order. Each `kernel-metadata.json` lists the datasets and earlier kernel outputs it needs.
   The test split is not read by any revision notebook (29-37).
3. Run the analysis scripts with `python -I <script>`.

Known limits:

- Several analysis scripts still resolve inputs relative to the original project layout
  (`01_Draft_Terbaru/evidence/...`, `02_Hasil_Eksperimen/runs/...`) or to `/kaggle/input`.
  `docs/PATH_MAP_old_to_new.tsv` and `docs/SOURCE_MANIFEST_old_repo.tsv` give the original location of each file.
- The repository has not been tested end to end from a fresh clone.
- Kernel 32 (CycleGAN-Turbo) did not complete on the available GPU; it is kept to document the attempt.
- The candidate screening used a vision-language agent on 3 October 2026; it cannot be rerun identically, so the
  baselines were compared under ranking-only selection (kernels 33, 36, 37).

## Licence

Code: MIT, see `LICENSE`. The vendored CycleGAN code in `code/mrtgan/vendor/pytorch_cyclegan` keeps its BSD
licence. `ultralytics` is a dependency under AGPL-3.0 and is not included. Released images, masks and weights:
CC BY 4.0, with attribution to the source dataset.

## Citation

See `CITATION.cff`. Correspondence: Tri Arief Sardjono (sardjono@bme.its.ac.id).
