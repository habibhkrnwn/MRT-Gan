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
| Inference scripts of the two diffusion baselines | `code/baselines/` |
| Kaggle notebooks with their `kernel-metadata.json` (GPU runs) | `notebooks/` |
| Local scripts that compute the tables and figures | `analysis/` |
| Old-to-new path map and SHA-256 manifest of copied files | `docs/` |
| Generator weights, all generated candidates, fruit masks, split identifiers, detector-run archives | GitHub Release of this repository (see below) |
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
| `candidates.zip` | All 217 generated candidates per generator and ablation (Table 2) |
| `candidates_inpaint_baseline.zip`, `candidates_anchor_baseline.zip` | Candidates of the two diffusion baselines (Table 2) |
| `masks_and_splits.zip` | Training fruit masks, crop manifest, colour statistics, photograph split, capture-group and near-duplicate lists |
| `S2_k_paste_real_runs_no_weights.zip` (kernel 30) | Every detector run of session S2 without the `.pt` weights: per-image validation predictions (`predictions.coco.json`), ground truth, arm manifests, loss and validation histories, run summaries |
| `revision_run_logs.zip` | Training logs of kernels 30, 31, 33, 34, 35, 36 and 37: detector chains, LoRA training and AnchorDiffusion editing |
| `SHA256SUMS` | Checksums; verify with `sha256sum -c SHA256SUMS` |

The released images and masks are derived from the Puspalebo dataset and keep its CC BY 4.0 licence.

YOLOv8n detector weights are not part of the release because of their size (about 1.4 GB per detector
session); every detector run is documented by its logs, validation history and per-image predictions, and the
weights are available from the corresponding author on request.

## Evidence

`evidence/` holds the small files behind the reported numbers, copied from the Kaggle kernel outputs:

| Folder | Contents |
|---|---|
| `splits_masks/` | `split_ids.csv` (every photograph with its group, capture session, split, exclusion flag and SHA-256), `groups.csv`, `near_duplicates.csv`, `crops_manifest.json`, and the colour-centroid statistics and margin calibration (`color_centroids.json`, `color_margin.json`). The mask images themselves are in `masks_and_splits.zip` in the Release. |
| `screening/ranking/<session>/` | Ranking of candidates by the summed absolute log ratio over five texture measures (`rank`, `used`, `parent_id`, `distance`, the five measures). `ranking_MRTGAN_s12.csv` and `ranking_colour_transfer.csv` list the accepted candidates and the 22 used; `ranking_unscreened_*` list all candidates for the ranking-only baseline comparison. The MRT-GAN rankings of the sessions agree up to the fifth decimal. |
| `predictions/<session>/paste_log.json` | Scene photograph and source fruit of every added image (`paste_real` control, Table 5 scene counts) |
| `predictions/S2_k_paste_real/runs/` | Arm manifests (which images each arm added), `results.json` per group, and `run_summary.json`, `validation.json` and `identity.json` per arm and seed; the full runs with per-image predictions are in `S2_k_paste_real_runs_no_weights.zip` in the Release |
| `predictions/<session>/*_result.sha256` | SHA-256 of the original Kaggle result archive of each detector session; the archives of the other sessions (with weights and per-image predictions) are available on request |
| `baselines/INPAINT`, `baselines/ANCHOR` | Settings (`summary.json`), training items and per-seed candidate records with parent IDs for the two diffusion baselines |
| `checkpoints/r512_ablation/` | SHA-256 of the matched 512-pixel ablation archive (kernel 29) |
| `kernel_logs/` | Dated Kaggle run logs of kernels 29-37 |

Sessions: `S1_main_arms` (kernels 27, 28), `S2_k_paste_real` (30), `heldout_dsc_folds` (31),
`S3_inpainting_baseline` (33), `S4_anchordiffusion_baseline` (36, and 37 in `heldout_folds/`).
None of the revision kernels read the test split.

The per-candidate decisions of the screening agent, the prompts as sent, the farmer ratings and the blinding keys
are not in this repository; they are available to the editor and reviewers from the corresponding author on
request.

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
