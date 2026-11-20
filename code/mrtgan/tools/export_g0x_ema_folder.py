"""Build G0X_EMA/: everything about the G0X-EMA generator and nothing about other generators.

    python tools/export_g0x_ema_folder.py --out G0X_EMA

Included: G0X-EMA config and data policy, the code needed to reproduce it, the
notebooks that ran it, the G0X-EMA generator run (checkpoints, losses, sweeps,
raw and EMA bundles with their blind reviews), the 9-parent detector arms of
G0X-EMA with their real_only / geometry_balanced controls, the whole full-data
balancing experiment (G0X-EMA synthetic, oversampling and real_only controls,
pilot, lock, one-time test evaluation), the split audit and the pair figure.

Excluded: outputs of G0, G4, G0X-v2 and the short proxy runs. Files that mix
G0X-EMA with another generator (the Kaggle logs of notebooks 08 and 09, the
merged EMA review sheet and the group-0 detector results) are not copied; a
derived G0X-only file is written instead and named `*_g0x_only.json`.

Run artifacts are hard links to `runs/`, so they cost no extra disk space; do not
edit them in place. Code, notebooks, configs and documents are real copies, so
later edits of the main code do not change this snapshot.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
BENCH = ROOT.parent / "14_Benchmark_Kaggle"

TOOLS = ["build_crops.py", "prepare_g0x_data.py", "calibrate_color_margin.py", "train_cyclegan.py",
         "generate_candidates.py", "run_qc.py", "review.py", "merge_review.py", "checkpoint_sweep.py",
         "build_arms.py", "train_detector_arms.py", "generate_balance_candidates.py", "build_balance_arms.py",
         "build_review_controls.py", "lock_final.py", "final_test_eval.py", "audit_split_leakage.py",
         "package_bundles.py", "export_g0x_ema_folder.py"]
CONFIGS = ["g0x_ema.json", "g0x_data_policy.json"]
NOTEBOOKS = ["08_cyclegan_ema.ipynb", "09_cyclegan_ema_detector.ipynb", "10_detector_convergence_pilot.ipynb",
             "11_balancing_detector.ipynb"]
DETECTOR_ARMS = ["real_only", "geometry_balanced", "cyclegan_g0x_ema_raw"]


def link_tree(src, dst, copy=False):
    """Hard-link (or, with `copy`, copy) every file of `src` under `dst`."""
    src, dst = Path(src), Path(dst)
    for path in sorted(src.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copy2(path, target)
            continue
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)


def link(src, dst, copy=False):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def g0x_only_notebook(source, out):
    """Notebook 08 reduced to the G0X-EMA job; for re-running only, never executed as such."""
    nb = json.loads(Path(source).read_text())
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        text = text.replace("JOBS = [('config/g0_ema.json', 'melon_cyclegan_g0_ema', SOURCE_CROPS),\n        ", "JOBS = [")
        text = text.replace("BUDGET = {'melon_cyclegan_g0_ema': 240, 'melon_cyclegan_g0x_ema': 240}",
                            "BUDGET = {'melon_cyclegan_g0x_ema': 240}")
        if cell["cell_type"] == "markdown" and "EMA of generator weights" in text:
            text = ("# CycleGAN melon: G0X-EMA only\n\nReduced copy of notebook 08 that trains only G0X-EMA "
                    "(`config/g0x_ema.json`, curated 298-crop pool, 60+60 epochs, AMP, EMA 0.999), then writes "
                    "the checkpoint sweeps and the raw and EMA bundles. The published G0X-EMA run came from "
                    "notebook 08, which trained G0-EMA on the second GPU at the same time. This reduced "
                    "notebook has not been executed.\n")
        cell["source"] = text.splitlines(keepends=True)
    Path(out).write_text(json.dumps(nb, indent=1) + "\n")


def detector_g0x_only(out):
    base = RUNS / "kaggle_ema_detector/detector_cyclegan_ema"
    combined = json.loads((base / "results.json").read_text())
    seeds = combined["seeds"]
    results = {k: v for k, v in combined["results"].items() if k.split("@")[0] in DETECTOR_ARMS}
    rows = [r for r in combined["summary"] if r["arm"] in DETECTOR_ARMS]
    pairs = {}
    for a, b in (("cyclegan_g0x_ema_raw", "real_only"), ("cyclegan_g0x_ema_raw", "geometry_balanced"),
                 ("geometry_balanced", "real_only")):
        diffs = [results[f"{a}@{s}"]["map50_95"] - results[f"{b}@{s}"]["map50_95"] for s in seeds]
        pairs[f"{a} - {b}"] = dict(mean=statistics.fmean(diffs), sd=statistics.pstdev(diffs),
                                   wins=sum(d > 0 for d in diffs), per_seed=diffs)
    derived = dict(derived_from="runs/kaggle_ema_detector/detector_cyclegan_ema/results.json (arm cyclegan_g0_ema_raw removed)",
                   arms=DETECTOR_ARMS, seeds=seeds, updates=combined.get("updates"), scope=combined.get("scope"),
                   results=results, summary=rows, paired_map50_95=pairs, test_used=combined["test_used"])
    Path(out).write_text(json.dumps(derived, indent=2) + "\n")
    return derived


def f(x, digits=4):
    return "—" if x is None else f"{x:.{digits}f}".replace(".", ",")


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def write_readme(out):
    """Folder map and G0X-EMA data only; every number read from files inside the folder."""
    out = Path(out)
    gen = out / "runs/01_generator_g0x_ema"
    loss = json.loads((gen / "checkpoints/melon_cyclegan_g0x_ema/loss.json").read_text())
    cfg = json.loads((out / "code/cyclegan/config/g0x_ema.json").read_text())
    tail = [r for r in loss["history"] if r["epoch"] > max(h["epoch"] for h in loss["history"]) - 10]
    keys = ["D_A", "D_B", "G_A", "G_B", "cycle_A", "cycle_B", "idt_A", "idt_B", "color_margin_A", "color_margin_B"]
    parts = ["# G0X-EMA", "",
             "Folder khusus generator CycleGAN **G0X-EMA** dan evaluasinya. Dibangun oleh "
             "`code/cyclegan/tools/export_g0x_ema_folder.py`. Artefak run berupa hard link ke `Ujicoba CycleGAN/runs/` "
             "(jangan diedit di tempat); kode, notebook dan dokumen berupa salinan terpisah. "
             "Isi berupa data dan statistik, tanpa analisis. Output generator lain (G0, G4, G0X-v2, run pendek) tidak "
             "disertakan. Log Kaggle notebook 08 dan 09 serta lembar review gabungan EMA tidak disalin karena memuat G0; "
             "`runs/02_detector_9_parents/results_g0x_only.json` adalah turunan tanpa arm G0.", "",
             "## Struktur", "",
             table(["Path", "Isi"], [
                 ["`code/cyclegan/`", "paket `melon_cyclegan`, `vendor/pytorch_cyclegan`, tools, config `g0x_ema.json` dan `g0x_data_policy.json`"],
                 ["`code/14_Benchmark_Kaggle/`", "`src/`, `tools/` (trainer detektor), `make_review_tool.py`"],
                 ["`notebooks/`", "notebook 08 (generator, juga melatih G0-EMA di GPU kedua), 09 (detektor 9 induk), 10 (pilot), 11 (balancing); `08b_g0x_ema_only_not_run.ipynb` = versi 08 hanya G0X-EMA, belum dijalankan"],
                 ["`runs/01_generator_g0x_ema/`", "checkpoint (G raw dan EMA tiap 10 epoch, D latest), loss, log, sapuan checkpoint raw/EMA, bundle raw/EMA dengan review, metadata pool 298 crop"],
                 ["`runs/02_detector_9_parents/`", "run detektor arm `cyclegan_g0x_ema_raw`, kontrol `real_only` dan `geometry_balanced`, hasil turunan"],
                 ["`runs/03_balancing_full_train/`", "kandidat sintetik full frame, kontrol review, review gabungan, pilot, detektor 7 arm, lock, evaluasi test final, audit split"],
                 ["`figures/`", "10 pasang foto nyata ripe dan sintetik unripe (pemilihan acak seed 2026)"],
                 ["`BALANCING_PLAN.md`", "pra-registrasi dan log keputusan eksperimen balancing"],
             ]), "",
             "Reproduksi lokal: jalankan dari `code/cyclegan/` dengan `MELON_DATASET` menunjuk dataset beku regrouped v4 "
             "(manifest `185e1422…`). Contoh: `python tools/train_cyclegan.py --config config/g0x_ema.json --dataroot <pool 298> "
             "--checkpoints-dir <dir>`; pool dibuat dengan `tools/build_crops.py` lalu `tools/prepare_g0x_data.py`.", "",
             "## 1. Konfigurasi dan training generator", "",
             table(["Hal", "Nilai"], [
                 ["Config", "`g0x_ema.json`"], ["netG / netD", f"{cfg['netG']} / {cfg['netD']}"],
                 ["Jadwal", f"{cfg['n_epochs']}+{cfg['n_epochs_decay']} epoch, lr {cfg['lr']}, batch {cfg['batch_size']}"],
                 ["Loss", f"lambda_A/B {cfg['lambda_A']}/{cfg['lambda_B']}, identity {cfg['lambda_identity']}, color margin {cfg['lambda_color_margin']} (m {cfg.get('color_margin', 'dari color_margin.json')}), composite mask {cfg['composite_mask']} (dilate {cfg['mask_dilate']}, feather {cfg['mask_feather']}), DiffAugment `{cfg['diffaug']}`"],
                 ["Presisi / EMA", f"AMP {cfg['amp']} / decay {cfg['ema_decay']}"],
                 ["Seed", cfg["seed"]], ["Pool", "298 crop (217 ripe, 81 unripe), digest `8cec926a…`"],
                 ["Epoch selesai / berhenti dini", f"{loss['epochs']} / {loss['stopped']}"],
                 ["Menit latih (Tesla T4)", f(loss['seconds'] / 60, 1)],
             ]), "",
             "Loss rata-rata 10 epoch terakhir (dibobot):", "",
             table(keys, [[f(statistics.fmean(r[k] for r in tail), 3) for k in keys]]), ""]
    # QC and review of the two bundles.
    rows_qc, rows_rv = [], []
    for weights in ("raw", "ema"):
        b = gen / f"bundle_cyclegan_g0x_ema_{weights}"
        q = json.loads((b / "quality_control.json").read_text())
        recs = json.loads((b / "records.json").read_text())
        for d in ("ripe_to_unripe", "unripe_to_ripe"):
            rr = [r for r in q["rows"] if r["direction"] == d]
            rows_qc.append([weights, d, len(rr), sum(r["classifier_pass"] for r in rr), sum(r["color_moves_to_target"] for r in rr),
                            f(statistics.fmean(r["ssim_inside"] for r in rr), 3), f(statistics.fmean(r["ssim_outside"] for r in rr), 3),
                            f(statistics.fmean(r["partial_translation_share"] for r in rr), 3)])
            rv = [r for r in recs if r["direction"] == d]
            rows_rv.append([weights, d, f"{sum(r['status'] == 'accepted' for r in rv)}/{len(rv)}"])
        parents = {r["parent_id"] for r in recs}
        accepted = {r["parent_id"] for r in recs if r["status"] == "accepted"}
        rows_rv.append([weights, "induk dengan ≥1 diterima", f"{len(accepted)}/{len(parents)}"])
    parts += ["## 2. Bundle 9 induk (checkpoint final): QC dan review buta", "",
              table(["Bobot", "Arah", "n", "Classifier lolos", "Bergerak ke target", "SSIM dalam", "SSIM luar", "Partial translation"], rows_qc), "",
              table(["Bobot", "Arah / cakupan", "Diterima"], rows_rv), ""]
    # Sweeps.
    rows_sw = []
    epochs = None
    for weights in ("raw", "ema"):
        sw = json.loads((gen / f"sweep_g0x_ema_{weights}.json").read_text())
        epochs = [r["epoch"] for r in sw["epochs"]]
        rows_sw.append([weights] + [f(r["per_direction"]["ripe_to_unripe"], 3) for r in sw["epochs"]])
    parts += ["## 3. Partial translation ripe→unripe per checkpoint", "", table(["Bobot"] + [str(e) for e in epochs], rows_sw), ""]
    # Detector 9 parents.
    det = json.loads((out / "runs/02_detector_9_parents/results_g0x_only.json").read_text())
    parts += ["## 4. Detektor, scope 9 induk (valid)", "",
              table(["Arm", "mAP50:95", "SD", "AP unripe", "Recall unripe"],
                    [[r["arm"], f(r["map50_95"]), f(r["sd"]), f(r["ap_unripe"]), f(r["recall_unripe"])] for r in det["summary"]]), "",
              table(["Pasangan", "Rata-rata selisih", "SD", "Menang"],
                    [[k, f(v["mean"]), f(v["sd"]), f"{v['wins']}/5"] for k, v in det["paired_map50_95"].items()]), ""]
    # Balancing.
    bal = out / "runs/03_balancing_full_train"
    res = json.loads((bal / "detector_7_arms/balance_detector/results.json").read_text())
    test = json.loads((bal / "final_test/test_results.json").read_text())
    seeds = res["seeds"]
    rows_b = []
    tsum = {r["arm"]: r for r in test["summary"]}
    for arm in res["arms"]:
        ks = [f"{arm}@{s}" for s in seeds]
        rows_b.append([arm, res["arms_summary"]["arms"][arm]["total_per_class"]["1"],
                       f(statistics.fmean(res["results_selected"][k]["map50_95"] for k in ks)),
                       f(statistics.fmean(res["results_final"][k]["map50_95"] for k in ks)),
                       f(tsum[arm]["map50_95"]), f(tsum[arm]["sd"]), f(tsum[arm]["ap_unripe"]), f(tsum[arm]["recall_unripe"])])
    comps = [[k, "utama" if c["primary"] else "sekunder", f(c["map50_95"]["mean"]), f(c["map50_95"]["sd"]),
              f"{c['map50_95']['wins']}/{c['map50_95']['n']}", " – ".join(f(v) for v in test["bootstrap"]["pairs"][k]["ci95"])]
             for k, c in test["comparisons"].items()]
    cand = json.loads((bal / "candidates_g0x_ema/records.json").read_text())
    parts += ["## 5. Eksperimen balancing, train penuh (301 foto nyata: 217 ripe / 84 unripe)", "",
              f"Kandidat sintetik G0X-EMA raw: {len(cand)}; diterima review {sum(r['status'] == 'accepted' for r in cand)}. "
              f"Rasio terpilih dari valid: `{test['selected_ratio_arm']}`. Test: {test['images']} citra, satu kali "
              f"({test['started_at']}).", "",
              table(["Arm", "Unripe total", "Valid terpilih", "Valid final", "Test mAP50:95", "Test SD", "Test AP unripe", "Test recall unripe"], rows_b), "",
              table(["Pasangan test", "Status", "Rata-rata selisih", "SD", "Menang", "IK 95% bootstrap"], comps), ""]
    (out / "README.md").write_text("\n".join(parts) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "G0X_EMA"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first to rebuild")

    # Code snapshot, same layout as the Kaggle code dataset so melon_cyclegan.config resolves.
    code = out / "code/cyclegan"
    link_tree(ROOT / "melon_cyclegan", code / "melon_cyclegan", copy=True)
    link_tree(ROOT / "vendor", code / "vendor", copy=True)
    for name in TOOLS:
        link(ROOT / "tools" / name, code / "tools" / name, copy=True)
    for name in CONFIGS:
        link(ROOT / "config" / name, code / "config" / name, copy=True)
    link(ROOT / "requirements.txt", code / "requirements.txt", copy=True)
    for sub in ("src", "tools"):
        link_tree(BENCH / sub, out / "code/14_Benchmark_Kaggle" / sub, copy=True)
    link(BENCH / "make_review_tool.py", out / "code/14_Benchmark_Kaggle/make_review_tool.py", copy=True)

    # Notebooks as run, plus a G0X-only reduction of notebook 08.
    for name in NOTEBOOKS:
        link(ROOT / "notebooks" / name, out / "notebooks" / name, copy=True)
    g0x_only_notebook(ROOT / "notebooks/08_cyclegan_ema.ipynb", out / "notebooks/08b_g0x_ema_only_not_run.ipynb")

    # 1. Generator run.
    gen = out / "runs/01_generator_g0x_ema"
    link_tree(RUNS / "kaggle_ema/checkpoints/melon_cyclegan_g0x_ema", gen / "checkpoints/melon_cyclegan_g0x_ema")
    link(RUNS / "kaggle_ema/melon_cyclegan_g0x_ema.log", gen / "melon_cyclegan_g0x_ema.log")
    for weights in ("raw", "ema"):
        link(RUNS / f"kaggle_ema/sweep_g0x_ema_{weights}.json", gen / f"sweep_g0x_ema_{weights}.json")
        link_tree(RUNS / f"kaggle_ema/bundle_cyclegan_g0x_ema_{weights}", gen / f"bundle_cyclegan_g0x_ema_{weights}")
    link_tree(RUNS / "kaggle_ema/crops256_g0x_v2", gen / "training_pool_crops256_g0x_v2")

    # 2. Detector, 9 parents: G0X-EMA arm and its controls.
    det = out / "runs/02_detector_9_parents"
    base = RUNS / "kaggle_ema_detector/detector_cyclegan_ema"
    for arm in DETECTOR_ARMS:
        for run_dir in sorted(base.glob(f"group*/{arm}__seed*")):
            link_tree(run_dir, det / run_dir.name)
    link(RUNS / "kaggle_ema_detector/detector_group1.log", det / "detector_group1_g0x_and_geometry.log")
    detector_g0x_only(det / "results_g0x_only.json")

    # 3. Balancing experiment (all arms use G0X-EMA synthetic or are controls).
    bal = out / "runs/03_balancing_full_train"
    link_tree(RUNS / "balance_candidates_g0x", bal / "candidates_g0x_ema")
    link_tree(RUNS / "review_controls_balance", bal / "review_controls")
    link_tree(RUNS / "review_merged_balance", bal / "review_merged")
    link_tree(RUNS / "kaggle_pilot", bal / "pilot_convergence")
    link_tree(RUNS / "kaggle_balance", bal / "detector_7_arms")
    link(RUNS / "FINAL_LOCK_balance.json", bal / "FINAL_LOCK_balance.json")
    link_tree(RUNS / "final_test_balance", bal / "final_test")
    link(RUNS / "final_test_balance.log", bal / "final_test.log")
    link(RUNS / "split_leakage_audit.json", bal / "split_leakage_audit.json")
    link(ROOT / "BALANCING_PLAN.md", out / "BALANCING_PLAN.md", copy=True)

    # Figure.
    for suffix in ("png", "json"):
        link(RUNS / f"figures/pasangan_real_sintetik_g0x_ema.{suffix}", out / f"figures/pasangan_real_sintetik_g0x_ema.{suffix}")
    write_readme(out)
    print("exported ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
