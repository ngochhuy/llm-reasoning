# Báo cáo task 2.9 — Chuẩn bị công cụ profiling KV cache

## Nội dung đã hoàn thành

Thêm `src/llm_reasoning/profiling.py`, gọi bằng `python -m llm_reasoning.profiling`. Script nạp cùng checkpoint/tokenizer một lần, đo cache OFF và ON trên cùng model, dtype, prompt và số token mới. Không huấn luyện hoặc thay đổi file trọng số.

CLI nhận checkpoint thật hoặc tạo checkpoint random nhỏ bằng `--smoke`. Có thể chạy CPU hoặc CUDA, chọn dtype, số warmup, số lần đo và các độ dài prompt. Mặc định khi nạp checkpoint: prompt **128/512/1.024/2.048**, sinh **128 token**, **5 lần đo**, warmup 1 lần mỗi mode/mỗi độ dài. Batch cố định bằng 1.

Smoke nhanh mặc định dùng prompt 16/32, sinh 8 token và 2 lần đo để thử trên CPU. Có thể truyền đủ bốn độ dài tiêu chuẩn như lần kiểm chứng của task này.

## Các chỉ số và cách tính

| Chỉ số | Định nghĩa |
| --- | --- |
| `prefill_ms` | Thời gian forward prompt lần đầu, gồm tạo K/V tensor khi cache ON |
| `first_token_ms` | Từ bắt đầu forward prompt đến khi chọn và nối token đầu tiên, gồm kiểm tra logits hữu hạn |
| `next_token_ms` | Trung bình thời gian mỗi token từ token thứ 2 đến N trong một lượt |
| `total_ms` | Tổng thời gian sinh N token, gồm token đầu tiên |
| `tokens_per_second` | N / tổng thời gian, có tính prefill |
| `decode_tokens_per_second` | (N−1) / tổng thời gian các bước sau token đầu tiên |
| `baseline_allocated_bytes` | Bộ nhớ tensor PyTorch đang cấp phát trước khi bắt đầu lượt đo CUDA |
| `peak_allocated_bytes` | Peak bộ nhớ tensor CUDA trong lượt, gồm model, prompt, K/V và tensor tạm |
| `peak_extra_allocated_bytes` | Peak allocated trừ baseline allocated |
| `peak_reserved_bytes` | Peak bộ nhớ CUDA allocator giữ lại, không đồng nghĩa bộ nhớ tensor đang dùng |
| `cache_bytes` | Tổng kích thước tensor K/V thực tế, tách biệt peak VRAM |

Các scalar trong bảng summary là **trung vị qua các lượt đo thành công**, không gồm warmup hoặc OOM. `next_token_ms` là trung vị của giá trị trung bình trong từng lượt; JSON còn lưu độ trễ từng token để phân tích tiếp.

Trên CPU, các cột VRAM là `null` trong JSON, ô trống trong CSV và `N/A` trong báo cáo. Không ghi 0 hoặc lấy RAM CPU giả làm VRAM.

## Quy trình đo công bằng

1. Tạo prompt có đúng số token bằng cách lặp một đoạn văn thông thường và cắt theo ID token. Lưu toàn bộ prompt ID và SHA256 để tái lập. Đây là workload hệ thống tổng hợp, không đánh giá khả năng hiểu ngữ cảnh dài.
2. Đưa model/prompt lên device trước vùng thời gian; tokenization, nạp checkpoint, clone prompt và tạo đối tượng cache nằm ngoài thời gian sinh. Clone prompt vẫn được tính trong peak memory.
3. Warmup mỗi mode tại mỗi độ dài. Mỗi request có cache mới, không tái dùng K/V giữa các lượt.
4. Sinh cố định N token bằng greedy, **bỏ dừng theo EOS** để khối lượng công việc hai mode bằng nhau. Token cuối sinh ra chưa cần được forward; cache cuối chứa L+N−1 token.
5. Cache OFF tính lại toàn bộ prefix; cache ON prefill một lần rồi forward một token mỗi bước. Dùng chính `TransformerLM`/`KVCache` đã kiểm thử, không thay inference backend.
6. Đồng bộ CUDA quanh mốc thời gian; đo wall-clock gồm forward, kiểm tra logits vị trí cuối hữu hạn, argmax, nối token và chi phí đo/đồng bộ. Không gọi đây là kernel-only latency hoặc throughput tối đa của một engine tối ưu.
7. Reset peak memory trước từng lượt đo; không empty CUDA allocator giữa các lượt thành công. Reserved memory có thể phản ánh allocator đã warmup từ lượt trước.
8. Đảo thứ tự ON/OFF giữa các lần lặp để giảm thiên lệch thứ tự. Kiểm tra ID của toàn bộ N token sinh ra ở từng cặp đo thành công.

Nếu OOM trong warmup, cấu hình đó ghi `warmup_oom` và không cố chạy các lượt đo của mode đó. OOM trong một lượt đo được ghi riêng; summary là `partial_oom` hoặc `oom`, không âm thầm giảm batch/sequence. Không coi cặp thiếu số liệu là “đã khớp”; CLI trả mã 1 khi có cặp không so sánh được hoặc không khớp. File kết quả vẫn được xuất để review. Lỗi logits NaN/Inf dừng phép đo thay vì báo throughput không hợp lệ.

## Artifact xuất ra

- `results.json`: config, provenance checkpoint, SHA256, dtype, model size, GPU/CPU, runtime, git commit/dirty, prompt ID, mọi lượt đo kể cả warmup/OOM, độ trễ từng token và kiểm tra greedy.
- `runs.csv`: số liệu từng lượt đo.
- `summary.csv`: trung vị theo độ dài và mode, số lượt thành công/OOM.
- `profiling.png`, `profiling.pdf`: bốn biểu đồ throughput, TTFT, next-token latency và peak allocated VRAM, tạo bằng Matplotlib.
- `report.md`: báo cáo ngắn kèm checkpoint hash, hardware, dtype, số token và bảng số liệu.

Chỉ xuất vào thư mục mới; không ghi đè kết quả cũ. Log có `git.dirty` để tránh hiểu nhầm commit SHA đã phản ánh mọi thay đổi chưa commit.

## Kiểm chứng thực chạy trong task này

Đã chạy bằng checkpoint random nhỏ, 2 lớp/hidden 32/4 head/FFN 128, **34.080 tham số**, vocab 292, context 2.304. Chạy trên **CPU FP32 một thread**, đủ prompt 128/512/1.024/2.048, sinh **4 token**, warmup 1 và đo 2 lần mỗi mode/mỗi độ dài.

Tổng có **8 lượt warmup + 16 lượt đo**. Cả **8 cặp đo** đều có token ID giống nhau giữa cache ON/OFF. Không gặp OOM. Cột VRAM không có số liệu vì đây là CPU. Biểu đồ PNG đã được mở để kiểm tra nhãn và bố cục.

Kết quả cuối tại `artifacts/task_2_9_profile_final/`; checkpoint/tokenizer thử tại `artifacts/task_2_9_profile_verified_checkpoint/`. Lần tạo checkpoint và chạy đầu tiên dùng:

```powershell
.venv\Scripts\python.exe -m llm_reasoning.profiling --smoke --output artifacts/task_2_9_profile_verified --prompt-lengths 128 512 1024 2048 --new-tokens 4 --repetitions 2 --warmup 1
```

Sau đó nạp lại cùng checkpoint để kiểm tra bản export báo cáo cuối:

```powershell
.venv\Scripts\python.exe -m llm_reasoning.profiling --checkpoint artifacts/task_2_9_profile_verified_checkpoint/random_model.pt --tokenizer artifacts/task_2_9_profile_verified_checkpoint/tokenizer.json --output artifacts/task_2_9_profile_final --prompt-lengths 128 512 1024 2048 --new-tokens 4 --repetitions 2 --warmup 1
```

Thêm 7 trường hợp kiểm thử trong `tests/test_profiling.py`: đối chiếu token với generation thật, công thức throughput/latency/median, cache bytes, warmup OOM, partial OOM, NaN, validation cấu hình, CLI export và CUDA memory (các nội dung được gom trong 7 test).

Suite toàn bộ: **121 passed, 3 skipped trong 16,62 giây** trên CPU. Sau thay đổi định dạng báo cáo, kiểm thử CLI/export được chạy lại và đạt. Ba kiểm thử chưa chạy cần CUDA: FP16 train, FP16 resume và bộ đếm VRAM profiling.

## Lệnh dùng tiếp và giới hạn

Thử nhanh từ môi trường mới:

```powershell
uv run --group model --group profile python -m llm_reasoning.profiling --smoke --output artifacts/task_2_9_profile_new
```

Lệnh dự kiến trên Kaggle T4 sau khi có checkpoint pretrain:

```bash
python -m llm_reasoning.profiling --checkpoint /path/to/pretrain.pt --tokenizer /path/to/tokenizer.json --device cuda --dtype float16 --prompt-lengths 128 512 1024 2048 --new-tokens 128 --repetitions 5 --warmup 2 --output /kaggle/working/kv_profile
```

Trên Kaggle dùng PyTorch CUDA sẵn có và cài nhóm công cụ biểu đồ khi cần; không đổi sang PyTorch CPU chỉ để đồng bộ local lockfile. Dtype thực tế sẽ được ghi trong kết quả. Chạy sau pretrain, giữ cùng checkpoint cho cả hai mode và lưu output bền vững.

**Chưa có báo cáo profiling bàn giao chính thức:** phép đo hiện dùng model thử rất nhỏ, CPU và 4 token sinh. Không dùng throughput này để suy ra tốc độ train hoặc hiệu năng model 36M/T4. Chưa kiểm chứng CUDA memory bằng phần cứng thực. Script tạo số liệu thí nghiệm; người thực hiện vẫn cần xác nhận đúng checkpoint sau pretrain, T4 và protocol trước khi nộp.

Bước tiếp theo là **task 2.10: hoàn thiện notebook 03 và chuẩn bị 04–06**, tích hợp các module đã kiểm tra, checkpoint/resume, dữ liệu và hướng dẫn review trước run chính.
