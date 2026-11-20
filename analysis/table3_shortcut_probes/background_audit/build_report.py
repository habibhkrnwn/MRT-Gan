from pathlib import Path
import json,statistics,hashlib
OUT=Path(__file__).resolve().parent
G=OUT.parents[3]
def load(n):return json.loads((OUT/n).read_text())
a=load('integrity.json');p=load('probes.json');d=load('detector_results.json');h=load('historical_subsets.json');v=load('fresh_validator.json');s=load('fresh_validator_sensitivity.json');hs=load('hash_sensitivity.json')
assert len(d)==len(v['results'])==len(s['results'])==5
selected=[v['results'][str(i)]['selected']['map50_95'] for i in range(5)]
final=[v['results'][str(i)]['final']['map50_95'] for i in range(5)]
reduced=[s['results'][str(i)]['map50_95'] for i in range(5)]
error=max(abs(x[k]['map50_95']-x[k]['reported_map50_95']) for x in v['results'].values() for k in ['selected','final'])
assert error<1e-10
ms=lambda x:f'{statistics.mean(x):.6f} ± {statistics.pstdev(x):.6f}'
lines=['# Audit empiris kebutuhan augmentasi sintetis melon',
'\nTanggal: 4 Oktober 2026. Status: **DIJALANKAN — audit dan inferensi CPU; tidak ada training baru atau penggunaan GPU.**',
'\n## Kesimpulan',
'\nBukti saat ini **belum mendukung alasan bahwa ketimpangan 217:84 menyebabkan kelemahan detektor yang harus diatasi dengan data sintetis**. Baseline yang sudah dilatih berhasil direproduksi. Hasil lebih mendukung penelitian tentang validitas transformasi visual dan keandalan evaluasi pada benchmark yang jenuh serta terkonfounded sumber gambar.',
'\nBias latar belakang memang tersedia, tetapi tidak boleh disamakan dengan bukti bahwa YOLO hanya memakai latar belakang. Pada lima checkpoint, kotak buah cukup untuk mempertahankan keberhasilan semua foto validasi; ketika kotak buah ditutup, tidak ada prediksi kelas benar pada confidence ≥0,25. Ini diagnostik OOD, bukan pembuktian lengkap mekanisme keputusan.',
'\n## 1. Ruang lingkup dan integritas',
f'\n- Manifest resmi dan manifest kerja identik: `{a["manifest_sha256"]}`.',
'- Manifest masih 432 baris, termasuk DSC_0093 yang berstatus accepted di manifest lama. Audit mengikuti naskah dengan pengecualian eksplisit DSC_0093; manifest tidak diubah.',
'- Efektif: train 301 (217 matang/84 belum matang), valid 87 (62/25), test 43 (31/12). Hitungan test hanya dari metadata manifest.',
'- Semua 388 gambar train+valid berupa symlink yang tersambung ke sumber yang benar; SHA-256 gambar dan label cocok 388/388; satu anotasi per foto, kelas/dimensi valid.',
'- Tidak ada duplikat byte-identik lintas train–valid. Tidak ada sesi pada groups.csv yang melintasi split. Ini tidak membuktikan identitas buah biologis atau foto sumber tidak berulang.',
'- Gambar dan label test tidak dibuka. Tidak ada pemindahan, penghapusan, perubahan label, atau perubahan naskah. Cache validator disimpan dalam folder audit sendiri.',
'\n## 2. Reproduksi baseline dengan validator',
'\nBobot real-only lima seed di runs/19_mrtgan_detector dievaluasi kembali pada 87 foto validasi. Checkpoint tetap; tidak ada pemilihan ulang. CPU, Ultralytics 8.4.157, PyTorch 2.14.0+cpu, batch 8, rect=True, imgsz=640, conf=0.001, IoU NMS=0.7, max_det=300. Run historis memakai Ultralytics 8.4.163.',
'\n| Checkpoint | mAP50:95 rerata ± SD populasi (5 seed) |', '|---|---:|',f'| Terpilih menurut validasi historis | {ms(selected)} |',f'| Akhir pelatihan | {ms(final)} |',
f'\nSeluruh 10 nilai mAP berhasil direproduksi terhadap run_summary.json; selisih maksimum {error:.2g}. Jadi mAP sekitar 0,99 terkonfirmasi, bukan hanya angka yang disalin dari PDF. Pemilihan checkpoint tetap memakai validasi yang sama, sehingga hasil bukan estimasi generalisasi independen.',
'\n## 3. Respons detektor ketika bagian gambar ditutup',
'\nKotak ground truth dipakai hanya untuk intervensi diagnostik. Bagian yang dihapus diisi RGB 114; ukuran dan posisi objek tetap. Ukuran keberhasilan: prediksi dengan skor tertinggi berkelas benar, confidence ≥0,25, dan IoU terhadap kotak asli ≥0,50. Ini bukan mAP dan tidak menghitung semua kemungkinan false positive.',
'\n| Seed | Foto asli | Hanya kotak buah | Kotak buah ditutup | Median confidence setelah buah ditutup |','|---|---:|---:|---:|---:|']
for seed,x in d.items():
 z=x['fresh_selected'];lines.append(f'| {seed} | {z["original"]["localized_correct_iou050_score025"]}/87 | {z["fruit_box_only"]["localized_correct_iou050_score025"]}/87 | {z["background_only_box_removed"]["localized_correct_iou050_score025"]}/87 | {z["background_only_box_removed"]["median_score"]:.4f} |')
lines += ['\nFoto yang sama diulang pada lima seed: tetap **87 foto independen yang tersedia, bukan 435 foto**. Kotak buah masih mengandung sedikit konteks sekitar buah. Penutupan menimbulkan pergeseran distribusi; hasil tidak membuktikan ketiadaan penggunaan konteks pada gambar asli, dan tidak membuktikan pengenalan kematangan fisiologis.',
'\n## 4. Probe sederhana dan pengelompokan sesi',
'\nStandardScaler dan LogisticRegression(C=0.1, class_weight=balanced) di-fit pada train, kemudian dievaluasi di valid. Fitur latar berupa statistik RGB/Lab/HSV di luar kotak; versi margin 15% mengecualikan daerah yang lebih luas. Parameter tidak dituning berdasarkan hasil.',
'\n| Fitur | AUC train→valid | AUC OOF dalam DSC train, dipisahkan menurut sesi |','|---|---:|---:|']
for k,name in [('bg_colour','Warna latar'),('bg_margin15_colour','Warna latar, margin 15%'),('fruit_box_colour','Warna dalam kotak buah'),('geometry','Geometri kotak'),('dimensions','Dimensi gambar')]:
 lines.append(f'| {name} | {p[k]["train_to_valid"]["auc"]:.4f} | {p[k]["within_DSC_train_session_grouped_OOF"]["auc"]:.4f} |')
lines += ['\n- Semua valid matang berasal dari DSC dan semua valid belum matang dari oke. Aturan berbasis nama sumber saja memberi 87/87 kecocokan label; ini diagnosis confounding, bukan model yang layak digunakan.',
'- AUC latar 0,9981 tetap bertahan ketika margin pengecualian kotak diperbesar. Bias ini tidak semata-mata disebabkan piksel dekat tepi kotak.',
'- Pada evaluasi DSC train yang benar-benar terpisah sesi, warna kotak buah tetap informatif, sedangkan probe warna latar jauh lebih lemah.',
'- Kode audit lama menggunakan photograph_group (ID foto) untuk GroupKFold. Angka yang disebut within-DSC dahulu diambil dari subset prediksi OOF setelah model dilatih pada DSC+oke. Audit baru memakai session dari groups.csv serta fit dan score hanya pada DSC. Ini perubahan protokol yang lebih ketat, sehingga selisih angka tidak boleh diatribusikan ke satu faktor saja.',
'- Batas CV: 233 foto DSC train, hanya 16 belum matang pada 8 kelompok sesi; satu fold holdout hanya berisi matang. AUC yang dilaporkan adalah pooled OOF, bukan rerata AUC fold atau estimasi presisi tinggi. Session berasal dari aturan nomor berkas dan kelas, bukan metadata waktu terverifikasi. Tidak ada klaim signifikansi.',
'\n## 5. Kemiripan lintas split dan sensitivitas',
'\nAudit dHash 256-bit dengan ambang ≤26 pada crop kotak rapat menandai lima kandidat. Pemeriksaan panel menunjukkan kesamaan kuat pola bayangan/permukaan; identitas foto/ buah tetap perlu ditelusuri dari arsip sumber. Tidak ada penghapusan otomatis.',
'\n| Valid | Train terdekat | Jarak dHash crop rapat |','|---|---|---:|']
for x in a['near_duplicate_flags_threshold26']: lines.append(f'| {x["valid"]} | {x["train"]} | {x["hamming"]} |')
lines += ['\nKode audit lama ternyata memakai padding 10% di sekeliling kotak. Ketika definisi itu direplikasi, hasilnya tepat dua flag lama: oke6396–DSC_0103 (5 bit), oke7065–oke8184 (8 bit). Jadi perbedaan lima versus dua disebabkan pilihan crop; metode dalam naskah sebaiknya menjelaskan padding tersebut.',
f'\nSensitivitas post-hoc validator pada 82 foto, dengan kelima kandidat dikeluarkan hanya dari salinan evaluasi: mAP50:95 baseline terpilih **{ms(reduced)}**, dibanding **{ms(selected)}** pada 87 foto. Skor tetap mendekati batas atas. Hasil ini tidak membuktikan semua kemiripan lain sudah ditemukan atau dataset bebas kebocoran.',
'\n## 6. Hasil subset historis yang ditemukan',
'\nIni pembacaan ulang arsip, bukan training baru. Subset menggunakan tiga seed, 3.000 pembaruan dan validasi 87 foto yang sama; seluruh ID subset diverifikasi berasal dari train resmi.',
'\n| Data real | Matang / belum matang | mAP50:95 terpilih real-only (rerata ± SD) |','|---|---:|---:|']
for k,x in h.items():
 z=x['report']['selected']['summary']['real_only'];lines.append(f'| {x["n"]} | {x["class_counts"]["0"]} / {x["class_counts"]["1"]} | {z["map50_95"]:.6f} ± {z["sd"]:.6f} |')
lines += ['\nKedua kelas dikurangi bersama sehingga proporsi kelas hampir tetap. Eksperimen ini menguji keterbatasan jumlah data pada subset tersebut, bukan efek kausal ketimpangan kelas. Walaupun hanya 75 gambar, baseline tetap sekitar 0,99. Tidak sah memilih subset yang lebih kecil semata-mata agar metode usulan tampak menang; ukuran dan pemilihan subset harus ditetapkan menurut pertanyaan riset, direplikasi, dan seluruh hasil dilaporkan.',
'\n## 7. Ketidaksesuaian jalur metrik ekspor',
'\nMenghitung mAP dari predictions.coco.json menghasilkan nilai yang berbeda dari validation.json; audit historis sensitivity_k22.json sudah mencatat perbedaan ini. Pada 20 run, selisih maksimum sekitar 0,004078. Validator langsung pada bobot baseline berhasil mereproduksi nilai resmi secara tepat, sehingga perbedaan ini bukan alasan menolak baseline 0,99.',
'\nJalur export_predictions memakai model.predict per gambar dan batas ekspor tersendiri, sedangkan validasi memakai model.val. Penyebab parameter spesifik belum diisolasi. Sensitivitas berbasis ekspor harus dilaporkan terhadap baseline ekspor yang sama, bukan dicampur dengan baseline validator; urutan arm pada kedua jalur dapat berbeda. Klaim naskah tentang urutan arm perlu diperjelas. hash_sensitivity.json mempertahankan seluruh hasil ekspor sebagai diagnostik, bukan pengganti tabel performa resmi.',
'\n## 8. Implikasi untuk latar belakang',
'\n### Klaim yang didukung',
'\n1. Dataset memiliki ketimpangan kelas dan hubungan kuat antara sumber citra dengan label.',
'2. Baseline pada split ini sudah mendekati batas atas, termasuk pada subset 75 dan 150 gambar historis.',
'3. Bias latar tersedia, tetapi respons YOLO menunjukkan informasi di kotak buah tetap penting dan cukup pada intervensi yang diuji.',
'4. Karena itu, kualitas citra sintetis, validitas perubahan kelas visual, dan manfaat detektor perlu dievaluasi terpisah.',
'\n### Klaim yang belum didukung',
'\n- Ketimpangan 217:84 menyebabkan kegagalan kelas belum matang.',
'- Menyeimbangkan jumlah gambar akan meningkatkan deteksi.',
'- YOLO hanya memanfaatkan latar belakang.',
'- MRT-GAN lebih berguna daripada kontrol pada data baru atau kondisi lapangan.',
'\n### Usulan paragraf latar belakang berdasarkan bukti',
'\n> Dataset golden melon yang digunakan memiliki 217 citra matang dan 84 citra belum matang pada bagian pelatihan. Namun, ketimpangan jumlah tersebut belum disertai penurunan performa kelas minoritas yang teramati: evaluasi ulang detektor real-only menghasilkan mAP50:95 sebesar 0,9907, dan baseline historis dengan 75 citra tetap mendekati 0,99. Pada saat yang sama, warna latar belakang dapat memisahkan kelas validasi dengan AUC 0,9981 karena sumber citra berkaitan erat dengan label. Temuan ini memotivasi evaluasi augmentasi yang memisahkan kualitas transformasi tampilan kematangan dari manfaatnya untuk deteksi, serta memeriksa apakah benchmark menyediakan pengujian yang memadai atas manfaat tersebut. Penelitian tidak mengasumsikan bahwa penyamaan jumlah kelas dengan sendirinya meningkatkan performa.',
'\nParagraf ini adalah usulan baru; naskah utama belum diubah. Angka baru dan metode audit harus dijelaskan sebelum dipakai dalam artikel.',
'\n## 9. Eksperimen selanjutnya, bila tujuan tetap manfaat deteksi',
'\n1. **Prioritas data:** telusuri foto yang sangat mirip dan identitas buah; kumpulkan kedua kelas pada sesi/perangkat/latar yang tumpang tindih, dengan label visual dari petani. Jangan menamai pengujian ulang split lama sebagai test independen.',
'2. **Protokol murah sebelum generator baru:** bangun evaluasi internal yang menahan sesi/identitas, dengan kedua kelas terwakili di setiap fold; latih ulang real-only dari inisialisasi yang sama. Model yang sudah melihat foto tersebut tidak sah dipakai untuk klaim holdout baru.',
'3. **Uji kelangkaan minoritas secara eksplisit:** tetapkan beberapa jumlah foto belum matang sementara jumlah matang tetap, subset bertingkat yang distratifikasi sumber/sesi, beberapa seed subset dan seed pelatihan. Laporkan bahwa perubahan ini juga mengubah total ukuran data; tambahkan kontrol total-data bila hendak memisahkan pengaruh rasio dari jumlah.',
'4. **Kontrol intervensi:** pada tiap subset yang ditentukan sebelum melihat hasil, bandingkan real-only, oversampling, transfer warna, dan MRT-GAN dengan jumlah tambahan serta pembaruan optimizer sama. Generator/statistik/seleksi harus hanya memakai training fold terkait untuk klaim low-data menyeluruh.',
'5. **Kriteria hasil:** AP dan recall tiap kelas, salah kelas, false positive, ukuran efek berpasangan, serta ketidakpastian menurut kelompok data; jangan memakai jumlah seed sebagai jumlah foto independen.',
'6. **Prioritas komputasi:** hasil audit belum memberi alasan mendahulukan pelatihan GAN lebih lama. Pelatihan GPU berikut belum dijalankan; saldo/alokasi aktual perlu diperiksa sesuai aturan proyek sebelum eksekusi.',
'\n## 10. Artefak dan reproduksi',
'\n- `audit.py`, `integrity.json`, `train_valid_inventory.json`, `features.npz`, `probes.json`, `source_rule.json`: audit dan probe baru.',
'- `detector_audit.py`, `detector_results.json`: inferensi intervensi lima checkpoint.',
'- `validate_baseline.py`, `fresh_validator.json`: reproduksi validator langsung, lima checkpoint terpilih dan lima akhir.',
'- `hash_sensitivity.py`, `hash_sensitivity.json`, `near_duplicate_diagnostic.png`: kemiripan dan sensitivitas ekspor.',
'- `validate_sensitivity.py`, `fresh_validator_sensitivity.json`: sensitivitas validator 82 foto.',
'- `historical_subsets.json`: ringkasan arsip subset, bukan run baru.',
'- `PROTOCOL.md`, `source_hashes.json`, log per skrip: keputusan protokol, provenance dan keluaran eksekusi.',
'\nLingkungan: Python virtual environment dari `requirements.txt`. Jalankan skrip dari folder naskah; urutan: audit.py → detector_audit.py → hash_sensitivity.py → validate_baseline.py → validate_sensitivity.py → build_report.py. Set `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4`. Seluruh operasi lokal CPU. Test tidak diperlukan.',
'\nBatas keseluruhan: audit eksploratif pada dataset yang hasilnya telah diketahui; tanpa data eksternal, tanpa training baru, tanpa validasi kematangan fisiologis. Tidak ada jaminan bahwa eksperimen berikut akan menguntungkan MRT-GAN.'
]
(OUT/'LAPORAN_AUDIT.md').write_text('\n'.join(lines)+'\n')
summary=dict(selected_mean=statistics.mean(selected),selected_sd=statistics.pstdev(selected),final_mean=statistics.mean(final),validator_max_reproduction_error=error,sensitivity82_mean=statistics.mean(reduced),sensitivity82_sd=statistics.pstdev(reduced),test_images_read=0,new_gpu_minutes=0,new_training_runs=0,selected_checkpoint_inference_runs=5,validator_runs=15)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
files=[OUT/f for f in ['audit.py','detector_audit.py','hash_sensitivity.py','validate_baseline.py','validate_sensitivity.py','PROTOCOL.md','LAPORAN_AUDIT.md']]
files += [G/'runs/19_mrtgan_detector/mrtgan_detector_v1/result/results.json',G/'runs/09_texture_subset_detectors/kaggle_v1_output/texture_subset_detectors_v1/result/results.json']
for run in (G/'runs/19_mrtgan_detector/mrtgan_detector_v1/result').glob('group*/*__seed*'):
 for name in ['run_summary.json','validation.json','predictions.coco.json','groundtruth.coco.json']:
  files.append(run/name)
(OUT/'source_hashes.json').write_text(json.dumps({str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
print(json.dumps(summary,indent=2))
