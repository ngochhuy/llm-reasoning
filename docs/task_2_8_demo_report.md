# Báo cáo task 2.8 — Demo nghiệm thu lõi bằng một lệnh

## Kết quả

Đã thêm CLI `python -m llm_reasoning.demo`. Luồng demo tạo/nạp checkpoint, kiểm tra forward, rồi sinh token tự hồi quy qua chính model, tokenizer và generation engine của dự án. Không dùng mô hình hoặc inference engine đóng gói sẵn.

Hai cách chạy:

- `--smoke`: tạo tokenizer nhỏ và model trọng số ngẫu nhiên, lưu checkpoint rồi **nạp lại từ đĩa** qua cùng loader dùng cho checkpoint thật. Không gọi training loop.
- `--checkpoint ... --tokenizer ...`: nạp một checkpoint có sẵn, kiểm tra SHA256 tokenizer và kiến trúc/vocab trước khi inference. Không khởi tạo thay thế nếu file lỗi hoặc không tìm thấy.

Demo dùng một prompt. `--prompt-format question` mặc định thêm định dạng `Question: ...\n<think>\n` giống dữ liệu SFT; `raw` giữ nguyên prompt. Có thể nhập `--prompt` hoặc `--prompt-file` UTF-8 để tránh lỗi dấu/escape trên shell Windows.

Hỗ trợ greedy (`temperature=0`), temperature/top-p sampling, seed, `--cache`/`--no-cache`, giới hạn token, CPU hoặc CUDA. Không cắt prompt dài hơn context. Đầu ra JSON trên terminal hoặc file `--output` chứa:

- Loại checkpoint, giai đoạn, số update/token đã lưu, checkpoint/tokenizer SHA256.
- Kiến trúc, số tham số thực tế, vocab, device và phiên bản PyTorch.
- Shape logits, kiểm tra logits hữu hạn và thời gian forward.
- Prompt, số token prompt, token ID mới, văn bản phần mới sinh, lý do dừng.
- Cấu hình decoding và thời gian generation.

`--output` và thư mục tạo smoke phải chưa tồn tại để tránh ghi đè kết quả review trước. Mặc định demo dùng CPU một thread; đây không phải cấu hình hiệu năng train.

## Nhận diện checkpoint thử

`save_checkpoint` có trường provenance tùy chọn `checkpoint_kind`: `smoke`, `training`, `unclassified`. Smoke demo lưu rõ `smoke` và có **0 optimizer update**. Checkpoint cũ không có trường này được hiển thị `unclassified`, không tự suy ra đã train chính thức chỉ dựa vào tên file hoặc stage.

`load_model_artifact` trả model và metadata trong cùng một lần đọc checkpoint; loader model-only cũ vẫn giữ API. Nhãn `training` chỉ là phân loại của người tạo artifact, không xác minh chất lượng benchmark. Các smoke artifact task trước chưa gắn nhãn được xem là unclassified.

## Chạy bằng một lệnh

Trong môi trường dự án đã có PyTorch:

```powershell
uv run --group model python -m llm_reasoning.demo --smoke
```

Lệnh này dùng `artifacts/task_2_8_demo/`. Nếu thư mục đã có, chọn thư mục mới:

```powershell
uv run --group model python -m llm_reasoning.demo --smoke --smoke-dir artifacts/task_2_8_demo_new --max-new-tokens 8 --output artifacts/task_2_8_demo_new/demo.json
```

Nạp lại artifact đã kiểm tra trong phiên Python mới:

```powershell
uv run --group model python -m llm_reasoning.demo --checkpoint artifacts/task_2_8_demo_verified/random_model.pt --tokenizer artifacts/task_2_8_demo_verified/tokenizer.json --prompt "2 + 3 = ?" --max-new-tokens 8 --no-cache
```

Sampling:

```powershell
uv run --group model python -m llm_reasoning.demo --checkpoint artifacts/task_2_8_demo_verified/random_model.pt --tokenizer artifacts/task_2_8_demo_verified/tokenizer.json --temperature 0.7 --top-p 0.95 --seed 42 --max-new-tokens 8
```

## Kiểm chứng thực chạy

Đã chạy tạo artifact qua CLI, rồi nạp lại cùng checkpoint ở một lệnh Python khác với cache OFF. Artifact tại `artifacts/task_2_8_demo_verified/` gồm:

- `random_model.pt`: checkpoint ngẫu nhiên, chưa có update train.
- `tokenizer.json`: tokenizer thử, vocab thực tế 292.
- `demo.json`: demo cache ON.
- `demo_cache_off.json`: demo cache OFF trên cùng checkpoint.

Model thử có 2 lớp, hidden 32, 4 head, FFN 128, context 256, **34.080 tham số**. Với prompt mặc định, đầu vào sau định dạng có 15 token; logits shape `[1, 15, 292]` và đều hữu hạn. Sinh đúng 8 token, dừng theo giới hạn token; token ID ON/OFF khớp hoàn toàn. Trọng số random sinh các token xuống dòng lặp lại; đây không phải câu trả lời toán học và không phải lỗi luồng generation.

Thêm 11 trường hợp kiểm thử trong `tests/test_demo.py`: subprocess tạo/nạp artifact, prompt file tiếng Việt UTF-8, shape/finite forward, cache ON/OFF, sampling tái lập, raw prompt/budget 0, prompt vượt context, sai tokenizer, CLI sai tham số, từ chối ghi đè smoke và checkpoint cũ không có provenance.

Lệnh suite:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv\Scripts\python.exe -m pytest -q
```

Kết quả: **115 passed, 2 skipped trong 10,04 giây** trên CPU. Hai gate CUDA FP16/train-resume từ task trước vẫn chưa được kiểm tra vì môi trường không có CUDA. Lệnh demo CUDA chưa được chạy trong task này.

## Giới hạn và bước tiếp theo

Thời gian in ra là thời gian của một lượt demo, không có warmup/repetition kiểm soát và không phải báo cáo profiling T4. Không dùng số này để kết luận speedup cache hoặc ước tính thời gian train. Kiến trúc chính khoảng 36M và dữ liệu chính chưa được train trong task này.

Demo chưa có giao diện chat nhiều lượt hoặc các chiến lược majority voting/Best-of-N; các phần đó thuộc giai đoạn reasoning/demo cuối. Task 2.8 tập trung nghiệm thu code forward, nạp trọng số và sinh tự hồi quy.

Bước tiếp theo là **task 2.9: chuẩn bị script profiling KV cache ON/OFF**, kiểm tra bằng trọng số thử trước; báo cáo bàn giao chính thức vẫn đo trên checkpoint sau pretrain và GPU T4.
