# Báo cáo task 2.10 — Notebook kiểm thử và chuẩn bị run Kaggle

## Phạm vi đã làm

- Tạo bốn notebook: `03_model_sanity_checks`, `04_pretraining`, `05_kv_cache_profiling`, `06_math_sft`. Mã nguồn sinh notebook nằm ở `scripts/build_stage2_notebooks.py`, giúp sửa và tái tạo nhất quán.
- Notebook 03 kiểm tra các module lõi, thực hiện overfit dữ liệu tổng hợp và so sánh train liên tục với resume. Các phép này dùng mô hình rất nhỏ, độc lập với mô hình chính.
- Notebook 04 đọc shard pretrain uint16 bằng `PretrainBatches`, kiểm tra tokenizer hash, số token mỗi nguồn, kích thước block và cấu hình. `inspect` chỉ kiểm tra; `smoke` chạy tối đa hai update với cấu hình nhỏ; `production` dùng cấu hình mục tiêu, có checkpoint theo thời gian và checkpoint cuối phiên. `MAX_UPDATES_THIS_SESSION` cho phép dừng pilot ở biên update, rồi resume cùng lịch train.
- Notebook 05 xác minh provenance checkpoint pretrain, đo cache ON/OFF bằng cùng CLI task 2.9. Chế độ `official` dùng T4, FP16, prompt 128/512/1024/2048, 128 token mới, warmup 2, năm lần đo và xuất JSON/CSV/PNG/PDF. Checkpoint pilot chưa hoàn tất cần được review rõ nếu muốn dùng do dừng sớm theo thời gian.
- Notebook 06 chỉ dùng dữ liệu GSM8K, xác minh tokenizer hash, nạp *trọng số* checkpoint pretrain, tạo optimizer và scheduler SFT mới. Bắt đầu một epoch, lưu snapshot từng epoch và đánh giá validation; có đường resume riêng cho SFT, giới hạn tối đa 3 epoch và trần 8M token. Trần được kiểm tại biên update nên có thể vượt nhiều nhất một accumulation window.
- `training_data.py` bổ sung batch có thể truy cập theo chỉ số và fingerprint từ manifest, file dữ liệu, split, kích thước batch, độ dài, pad ID. `ResumableBatchStream` dùng fingerprint và cursor để từ chối resume khi dữ liệu hoặc cách chia batch đổi. Pretrain dùng mmap, SFT dùng offset JSONL; không nạp hết token vào RAM.
- `load_model_artifact` trả thêm `total_steps` để notebook 05/06 nhận biết checkpoint pretrain mới chạy pilot. Hai notebook chỉ cho phép dừng sớm nếu người thực hiện chủ động bật cờ `APPROVED_EARLY_STOP` sau review.

## Đường dẫn và thao tác trước Kaggle

1. Chạy notebook 01–02 để tạo dữ liệu **chính thức**, không dùng artifact dry run hiện tại. Lưu tokenizer, `pretrain/manifest.json`, shard, `sft_gsm8k/manifest.json` và JSONL vào Kaggle Dataset/Notebook Version.
2. Commit mã nguồn đã review; đặt `REPO_REF` thành SHA 40 ký tự trong từng notebook Kaggle. Chạy notebook 03, xem kết quả tests, toy overfit và resume. Kaggle cần PyTorch có CUDA, NumPy, TensorBoard, Matplotlib và pytest; giữ bản torch CUDA cài sẵn.
3. Gắn dữ liệu dưới `/kaggle/input/llm-reasoning-data/` hoặc sửa `DATA_ROOT`/`SFT_ROOT` theo đường thực tế. Notebook 04 `inspect` xác minh đầu vào; sau review đổi thành `production`. Lưu toàn bộ `/kaggle/working/pretrain_run` ra nguồn bền vững sau mỗi phiên.
4. Notebook 05 và 06 chạy trong phiên mới, gắn `latest.pt` của pretrain dưới `/kaggle/input/llm-reasoning-pretrain/`. Notebook 05 đo độc lập, không sửa trọng số. Notebook 06 lưu `/kaggle/working/sft_run` để resume hoặc chọn snapshot theo validation.
5. Khi resume notebook 04/06, gắn checkpoint và đúng dữ liệu vào phiên mới, đặt `RESUME_FROM`, giữ nguyên SHA, cấu hình và dataset; nếu giữ logs từ phiên trước, logs không được đi trước checkpoint. `run_manifest.json` ghi SHA mã, hash tokenizer, fingerprint dữ liệu, cấu hình và số bước. Chạy notebook 04 và 06 không phụ thuộc biến trong bộ nhớ của notebook khác.

## Kiểm thử và kết quả

- `tests/test_training_data.py`: đọc block uint16, batch SFT có mask/padding, cursor resume, fingerprint đổi khi file đổi; **2 passed**.
- `scripts/verify_stage2_notebooks.py --smoke --sanity`: chạy notebook 03 (84 passed, 2 skipped, toy overfit và resume chính xác), rồi 04→05→06 ở các tiến trình Python riêng với artifact dữ liệu dry run hiện có. Pretrain nhỏ: 2 update, 1.024 token đầu vào. Profiling cache ON/OFF: tất cả cặp greedy khớp. SFT nhỏ: 4 update, 838 token đầu vào; validation loss 8,128 trên checkpoint ngẫu nhiên/ít bước, lưu `epoch_1.pt`. Checkpoint và báo cáo thử của 04–06 được ghi vào thư mục tạm.
- Toàn bộ test suite: **123 passed, 3 skipped**; các skip cần CUDA, máy local đang dùng torch CPU. Chưa kiểm chứng toàn bộ notebook ở Kaggle T4; FP16 và VRAM chỉ được xác minh trong run T4 sắp tới.

## Điểm cần review trước task 3.1

- SHA commit được pin, cấu trúc Kaggle Dataset và đường `DATA_ROOT`/`SFT_ROOT` thực tế.
- Manifest dữ liệu chính thức đạt gần 80M FineWeb-Edu + 20M OpenWebMath token train, có validation shard; tokenizer hash dùng chung với GSM8K SFT.
- GPU/quota và tốc độ pilot 200–500 update; cân nhắc mốc dừng theo thời gian trước khi vào SFT. Không suy thời gian hay VRAM T4 từ smoke CPU.
- Notebook 07 đánh giá suy luận thuộc giai đoạn 4, chưa triển khai trong task này.

**Trạng thái:** hoàn thành phần chuẩn bị task 2.10; chưa có lần pretrain/SFT chính thức, chưa có profiling T4. Dừng để review trước task 3.1.
