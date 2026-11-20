# Báo cáo task 2.7 — Checkpoint và resume

## Kết quả triển khai

Thêm `checkpoint.py`, `data_stream.py`, script `scripts/task_2_7_smoke.py` và kiểm thử checkpoint. Cập nhật `Trainer` để dừng/lưu tại ranh giới kết thúc accumulation window, và `TrainingLogger` để nối log khi resume.

Checkpoint đầy đủ lưu:

| Nhóm | Nội dung |
| --- | --- |
| Model | State dict, kiến trúc, vocab thực tế, dtype, trạng thái train/eval từng module |
| Tối ưu | AdamW, scheduler, GradScaler, weight decay, tổng số bước dự kiến, stage |
| Tiến độ | Số update thử/thành công, input token và target token đã xử lý |
| Dữ liệu | Fingerprint, số micro-batch, seed, shuffle, epoch, cursor của batch kế tiếp, permutation và RNG riêng của stream |
| RNG toàn cục | Python, Torch CPU, Torch CUDA nếu có, NumPy nếu được cài |
| Kiểm chứng | Phiên bản format, SHA256 tokenizer, phiên bản Torch/CUDA |

Tokenizer được giữ ở file JSON riêng; khi nạp phải cung cấp file có SHA256 và vocab khớp. Không lưu KV cache inference trong checkpoint train.

## Ranh giới lưu và dữ liệu

Chỉ lưu sau khi hoàn thành cả nhóm gradient accumulation và xóa gradient. `Trainer.train_epoch(..., max_updates=N)` cho phép dừng tại một ranh giới hợp lệ mà không đổi tổng schedule của run. `on_update` được gọi sau mỗi window đã commit; có thể gắn `PeriodicCheckpoint` để lưu khi đã qua 30 phút, hoặc lưu thủ công khi kết thúc stage/phiên.

Không lưu giữa micro-batch khi còn gradient đang tích lũy. Nếu một window thất bại sau khi đã lấy dữ liệu, trainer yêu cầu restore checkpoint trước khi tiếp tục. Điều này tránh tiếp tục từ cursor đã tiến nhưng optimizer chưa cập nhật.

`ResumableBatchStream` nhận một Sequence micro-batch có thể truy cập bằng chỉ số. Stream không prefetch hoặc dùng worker: cursor chỉ tăng khi trainer lấy batch. Đổi epoch bằng `next_epoch()` sau khi đã dùng hết epoch cũ; shuffle dùng generator riêng được lưu đầy đủ. Window cuối epoch chưa đủ số micro-batch vẫn được cập nhật theo mẫu số thực tế, như task 2.6.

Fingerprint phải đại diện cho dữ liệu bất biến và cách batching/collation. Caller cần xây từ manifest/checksum, revision, thứ tự shard, tokenizer và cấu hình batching; chỉ cùng số lượng batch là chưa đủ. Script thử tạo SHA256 từ nội dung records. Các adapter đọc shard/JSONL lớn và nối vào notebook sẽ được tích hợp ở các task sau; không giả định có thể resume chính xác một DataLoader bất kỳ với worker/prefetch.

## Lưu và nạp

`save_checkpoint` kiểm tra tokenizer/vocab, đúng stream đã đưa vào trainer và ranh giới update. Payload được chụp sang CPU rồi ghi file tạm trong cùng thư mục, flush/fsync và thay thế đường dẫn đích bằng `os.replace`. Nếu ghi lỗi, checkpoint cũ vẫn còn và file tạm được dọn. Đây là thao tác đồng bộ; có thời gian sao chép/ghi và cần thêm bộ nhớ CPU tạm.

`load_checkpoint` đọc trên CPU, kiểm tra format/hash tokenizer/fingerprint dữ liệu, dựng lại model và trainer từ config đã lưu, nạp optimizer/scheduler/scaler/counter/stream, rồi **khôi phục RNG cuối cùng** để việc khởi tạo model không làm lệch RNG của run. Không cho đổi stage hoặc schedule khi resume cùng run. Nạp lại vẫn giữ weight tying.

`load_model_checkpoint` chỉ nạp kiến trúc/trọng số và đặt eval, dùng cho inference hoặc chuyển pretrain → SFT. Khi chuyển stage, tạo Trainer SFT mới với optimizer/schedule SFT riêng trên trọng số pretrain đó; không tiếp tục optimizer pretrain.

Log CSV có thể nối bằng `TrainingLogger(directory, resume_step=saved_step)`. Các bước train mới phải tăng; nếu CSV đang chứa bước lớn hơn checkpoint thì báo lỗi, yêu cầu thư mục log segment mới thay vì ghi đè hoặc nối sai lịch sử. TensorBoard dùng `purge_step=resume_step+1`. Checkpoint không nhúng event files/CSV: phải bàn giao cùng artifact log.

## Kiểm thử thực chạy

Lệnh:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/task_2_7_smoke.py --output artifacts/task_2_7_smoke_verified
```

Suite cuối: **104 passed, 2 skipped trong 4,90 giây** trên CPU. Hai kiểm thử bị bỏ qua cần GPU CUDA: FP16 training từ task 2.6 và FP16 scaler/optimizer resume từ task 2.7.

Kiểm thử mới đối chiếu cả hai stage qua ranh giới epoch, bao gồm một window cuối epoch không đầy. Batch giả lập trong unit test cố ý sử dụng cả Python RNG, NumPy RNG và Torch RNG; sau restore, thứ tự dữ liệu, các giá trị ngẫu nhiên tiếp theo, loss/LR, trọng số, optimizer, scheduler, scaler và counter khớp chính xác. Ngoài ra kiểm tra sai tokenizer/dữ liệu/stage/format, model-only transfer, ghi lỗi vẫn giữ file cũ, periodic save, nối log và từ chối checkpoint giữa accumulation.

Script artifact dùng model nhỏ 1 lớp, hidden 8, 2 head, FFN 16, vocab giả 262, CPU FP32. Dữ liệu gồm 13 micro-batch giả, mỗi batch 6 token, accumulation 2. Mỗi stage chạy 10 update; so sánh chạy liên tục với ngắt sau update 3 rồi resume. Hai stage là hai bài kiểm thử độc lập, không phải run production hoặc các mô hình ứng viên để chọn theo test.

| Kết quả | Pretrain thử | SFT thử |
| --- | ---: | ---: |
| Update sau khi hoàn thành | 10 | 10 |
| Epoch (đánh số từ 0) | 1 | 1 |
| Cursor batch tiếp theo | 6 | 6 |
| Input token đã xử lý | 114 | 114 |
| Target token được tính loss | 95 | 76 |
| Sai khác tham số lớn nhất | **0** | **0** |
| Loss/LR, optimizer, scheduler, scaler, counter và data state | Khớp chính xác | Khớp chính xác |

Artifact: `artifacts/task_2_7_smoke_verified/summary.json`, tokenizer, các checkpoint `latest.pt` (tại bước ngắt) và `completed.pt` (kết thúc), CSV và TensorBoard cho run continuous/resumed.

## Cách tích hợp ở bước tiếp theo

```python
from llm_reasoning.checkpoint import PeriodicCheckpoint, save_checkpoint, load_checkpoint

# stream là ResumableBatchStream, đã có fingerprint dữ liệu + batching.
periodic = PeriodicCheckpoint('artifacts/checkpoints/latest.pt', stream,
                              tokenizer_path=tokenizer_path,
                              interval_minutes=trainer.config.checkpoint_interval_minutes)
trainer.train_epoch(stream, on_update=periodic)
# Lưu cuối stage/phiên, kể cả chưa đủ 30 phút.
save_checkpoint('artifacts/checkpoints/latest.pt', trainer, stream,
                tokenizer_path=tokenizer_path)

# Phiên mới: dựng lại cùng Sequence dữ liệu, seed/shuffle/fingerprint.
trainer = load_checkpoint('artifacts/checkpoints/latest.pt', stream,
                          tokenizer_path=tokenizer_path, device='cpu',
                          expected_stage='pretrain')
trainer.train_epoch(stream)
```

Đoạn cuối giả định đã tái tạo stream theo đúng dữ liệu; hàm load sẽ đặt lại cursor/order đã lưu. Nếu checkpoint ở cuối epoch, phải chuyển sang epoch kế tiếp trước khi train. Có thể chọn thư mục log segment mới và gắn `trainer.logger` sau khi load nếu log cũ đi trước checkpoint.

## Giới hạn

Exact resume hiện yêu cầu cùng loại device và cùng môi trường thực thi có tính xác định. Kiểm chứng bit-for-bit mới thực hiện trên CPU với cấu hình nhỏ; không cam kết giống bit giữa phần cứng/phiên bản Torch khác nhau. CUDA FP16 và trạng thái optimizer/scaler trên T4 vẫn là gate trước run chính.

Trên Kaggle, file trong working directory chưa tự trở thành artifact bền vững giữa các phiên: người thực hiện vẫn cần lưu Notebook Version/Output hoặc Dataset và kiểm tra nạp lại trước khi kết thúc phiên. Không thể bảo đảm mất điện/session bị ngắt đúng giữa ghi file vẫn lưu được update mới nhất; sẽ dùng checkpoint hợp lệ gần nhất.

Chưa chạy train model 36M trên dữ liệu chính. Bước tiếp theo là **task 2.8: demo nghiệm thu lõi**, nạp checkpoint thử, forward và sinh tự hồi quy qua một lệnh.
