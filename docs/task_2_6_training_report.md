# Báo cáo task 2.6 — Training loop và logging

## Thành phần đã xây

`src/llm_reasoning/training.py` cung cấp:

- `collate_sft`: right padding theo token pad, tạo riêng `attention_mask` cho token thật và `loss_mask` cho phần phản hồi.
- `target_mask` và `causal_lm_loss`: loss cross-entropy tự hồi quy, dịch logits/target và mask đúng chiều.
- `Trainer`: AdamW, gradient accumulation, clipping, FP16 autocast/GradScaler trên CUDA, lịch learning rate và validation.
- `TrainingLogger`: CSV và TensorBoard cho cả train/validation.
- `TrainState`: số bước thử, số optimizer update thành công, token đã xử lý và token được tính loss. Checkpoint/resume đầy đủ thuộc task 2.7.

## Loss và chuẩn hóa gradient

Logits vị trí `t` dự đoán token `t+1`. Pretrain tính loss trên các target hợp lệ; SFT chỉ tính loss nếu `loss_mask[t+1]=1`. Vì vậy token phản hồi đầu tiên vẫn được học từ logits ở vị trí prompt cuối. Prompt có thể nhận gradient qua attention/embedding; chỉ các target prompt bị loại khỏi loss.

Padding bị loại theo mask token thật. Attention hiện có causal mask nên padding bên phải không ảnh hưởng token thật ở phía trước. Không hỗ trợ left padding; không truyền response mask làm attention mask vì mô hình cần đọc prompt.

Mỗi micro-batch dùng loss dạng tổng. Gradient trong một accumulation window được chia cho **tổng số target được tính loss của toàn window**, không lấy trung bình các micro-batch mean. Kiểm thử xác nhận hai cách cập nhật tương đương khi micro-batch có số response token khác nhau.

Nếu cuối epoch còn ít micro-batch hơn cấu hình accumulation, vẫn cập nhật với mẫu số thực tế. Một window không có target được tính loss sẽ báo lỗi. Validation cũng dùng trung bình theo số target, không lấy trung bình loss của từng batch.

## Optimizer, schedule và precision

AdamW dùng learning rate theo stage: mặc định pretrain `3e-4`, SFT `3e-5`. Weight decay mặc định 0,1 cho tham số ma trận; RMSNorm và các tham số một chiều không decay. Weight tying không làm tham số bị thêm hai lần vào optimizer.

Warmup tuyến tính theo số optimizer update, dùng `ceil(warmup_fraction * total_steps)`. Sau warmup giảm cosine đến 10% learning rate đỉnh. Giá trị LR log là giá trị thực dùng cho update đó. Training loop sử dụng config mặc định warmup 5%, clip norm 1,0 trừ khi bài thử ghi khác.

FP16 yêu cầu CUDA: dùng autocast và GradScaler, unscale trước khi clipping. Nếu scaler bỏ qua update do overflow, scheduler không tiến; log đánh dấu `update_skipped`. `total_steps` là giới hạn số lần thử update, nên số optimizer update thành công có thể thấp hơn nếu overflow. CPU phải chọn FP32; không âm thầm chuyển FP16 về FP32.

Loss không hữu hạn gây lỗi rõ ràng. Với FP32, gradient không hữu hạn cũng gây lỗi. Không dùng KV cache trong train/validation loss.

Tạo `Trainer` riêng khi chuyển pretrain sang SFT để có optimizer/schedule mới trên cùng model. `train_epoch` nhận iterable batch do caller cung cấp; chưa tự đọc shard, tự lặp epoch, chọn checkpoint tốt nhất hoặc áp ngân sách SFT 1–3 epoch/8M token. Notebook/run orchestration ở task sau sẽ kết nối dữ liệu và các giới hạn này. Không gọi đây là lệnh train production hoàn chỉnh trước khi xong checkpoint/resume và review giai đoạn 2.

## Logging và số token

CSV ghi stage, train/validation, bước, loss, learning rate, gradient norm trước clipping, cờ skip, số token, thời gian và token/giây. TensorBoard ghi các scalar tương ứng. Thư mục đã có `metrics.csv` bị từ chối để tránh ghi đè run; cách mở log khi resume sẽ bổ sung ở task 2.7.

`processed_tokens` đếm input token thật, gồm prompt SFT, tính lại khi lặp dữ liệu; không gồm right padding. `supervised_tokens` chỉ đếm target được tính loss sau shift/mask. Thời gian mỗi window đồng bộ CUDA khi dùng CUDA, gồm forward/backward/optimizer và chuyển batch lên device, không gồm thời gian tải iterable hoặc ghi log sau update. Đây chưa phải phép đo profiling inference KV cache.

## Kiểm thử và bằng chứng thực chạy

Lệnh đã chạy:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/task_2_6_smoke.py --output artifacts/task_2_6_smoke_verified
```

Kết quả suite: **95 passed, 1 skipped trong 4,58 giây**. Kiểm thử bị skip là CUDA FP16/GradScaler vì môi trường không có CUDA; cần chạy trên T4 trước run chính. Nhánh scaler skip/scheduler được kiểm tra bằng scaler giả lập trên CPU, không thay thế kiểm chứng FP16 thực tế.

Các kiểm tra mới xác nhận shift/mask/padding, gradient accumulation tương đương batch lớn, flush window cuối, validation có trọng số token, warmup/cosine, clipping trước optimizer, đọc lại CSV/TensorBoard và overfit cho cả hai stage.

Bài thử artifact dùng model **2.368 tham số**, hidden 16, 1 lớp, vocab giả 17; một chuỗi giả 6 token, micro-batch 1, accumulation 2, learning rate 0,01, 60 update mỗi stage, seed 42, CPU FP32:

| Stage | Loss trước | Loss sau | Input token đã xử lý | Target tính loss |
| --- | ---: | ---: | ---: | ---: |
| Pretrain thử | 2,855988 | 0,074025 | 720 | 600 |
| SFT thử | 2,830411 | 0,067200 | 720 | 480 |

Hai stage trong smoke script là **hai bài kiểm thử độc lập**, mỗi bài khởi tạo model nhỏ từ cùng seed; không phải chuỗi train production pretrain → SFT và không tạo mô hình ứng viên để chọn theo benchmark. Validation trong bài overfit dùng chính batch thử để xác nhận khả năng ghi nhớ; không phải bằng chứng tổng quát hóa.

Artifacts thật ở `artifacts/task_2_6_smoke_verified/`: `summary.json`, `pretrain/metrics.csv`, `sft/metrics.csv` và TensorBoard event files của từng stage. Không commit artifact sinh ra vào Git.

## Cách xem và chạy lại

```powershell
uv sync --group model --group train --group dev
uv run --group model --group train pytest tests/test_training.py -q
uv run --group model --group train python scripts/task_2_6_smoke.py --output artifacts/task_2_6_smoke_new
uv run --group model --group train tensorboard --logdir artifacts/task_2_6_smoke_verified
```

Chọn thư mục output mới mỗi lần chạy smoke. Chưa chạy model 36M trên dữ liệu chính, chưa đo thời gian train/VRAM T4. Bước tiếp theo là **task 2.7: checkpoint/resume**, gồm optimizer, scheduler, scaler, RNG, vị trí dữ liệu, bước/token và kiểm thử run liên tục so với ngắt/resume.
