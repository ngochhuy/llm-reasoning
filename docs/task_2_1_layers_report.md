# Báo cáo task 2.1 — RMSNorm, RoPE và FFN GELU

## Phạm vi hoàn thành

Đã xây ba khối bằng PyTorch tại `src/llm_reasoning/layers.py`. Đây là thành phần của mô hình sẽ train từ đầu, chưa phải mô hình hoàn chỉnh hoặc checkpoint đã huấn luyện. Attention và ghép decoder thuộc task 2.2–2.3.

### RMSNorm

Chuẩn hóa chiều cuối theo `x / sqrt(mean(x²) + eps)`, sau đó nhân trọng số học được, khởi tạo bằng 1. Không có bias, mặc định `eps=1e-6`. Với FP16/BF16, tính thống kê trong FP32 để tránh tràn khi bình phương; đầu ra giữ dtype đầu vào. FP64 được giữ để kiểm tra gradient số.

### RoPE

Nhận tensor `[batch, heads, sequence, head_dim]`; xoay từng cặp tọa độ liền kề. Head dimension phải chẵn. Base mặc định 10.000, context mặc định 2.304 theo cấu hình mục tiêu. `offset` biểu diễn số token đã có trước đoạn hiện tại; kiểm tra offset không âm và không vượt context. Attention ở task sau phải dùng cùng quy ước cho Q và K.

Tần số dùng buffer chỉ số nguyên để không mất chính xác khi gọi `.half()`. Góc lượng giác được tính trong FP32 (FP64 với input FP64). Buffer không phải tham số học được và không cần lưu trong state dict; phải lưu cấu hình cùng checkpoint ở task 2.7.

### FFN GELU

Hai lớp linear không bias: `d_model → d_ff → d_model`, ở giữa là GELU chính xác. Với cấu hình mục tiêu 512/2048, FFN có 2.097.152 tham số mỗi lớp. RMSNorm hidden 512 có 512 tham số; RoPE không có tham số học được. Tổng tham số toàn mô hình sẽ đo ở task 2.3.

Pre-Norm sẽ được ghép trong decoder: `x + FFN(norm(x))`, và tương tự cho attention. Task này kiểm tra nhánh FFN theo công thức đó, chưa tạo decoder block trước task 2.3.

## Kiểm chứng

Đã cài nhóm dependency `model` chứa PyTorch và cập nhật `uv.lock`. Lần chạy kiểm thử dùng PyTorch 2.15.0 trên CPU, Python 3.11.

Lệnh đã chạy: `.venv\Scripts\python.exe -m pytest -q`.

Kết quả: **30 kiểm thử đạt trong 44,47 giây**, trong đó có 8 trường hợp kiểm thử cho task 2.1:

- RMSNorm đối chiếu công thức, input bằng 0, gradient số và gradient của trọng số.
- RoPE xử lý toàn chuỗi tương đương xử lý hai đoạn với offset; bảo toàn norm; vị trí 0 là phép đồng nhất.
- Đối chiếu hướng quay với giá trị sin/cos độc lập, tránh kiểm thử chỉ dựa vào cùng implementation.
- Kiểm tra gradient số RoPE.
- Đầu vào FP16/BF16/FP32: đầu ra và gradient hữu hạn, giữ dtype.
- Nhánh residual pre-norm FFN giữ shape, có gradient cho cả input và tham số, không có bias.
- Từ chối dimension và offset/context không hợp lệ.

`git diff --check` không phát hiện lỗi whitespace. Chưa đo hiệu năng GPU hoặc chạy train; các kiểm thử CPU không phải báo cáo profiling T4.

## Cách chạy lại và bước tiếp theo

```powershell
uv sync --group model --group dev
uv run --group model pytest tests/test_layers.py -q
```

Sau review, task 2.2 sẽ tự viết causal attention: Q/K/V, causal mask, softmax và output projection; kiểm tra token tương lai không ảnh hưởng token quá khứ. Task 2.1 chưa cần chạy notebook hay bật Kaggle GPU.
