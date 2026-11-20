# Báo cáo task 2.2 — Causal self-attention

## Nội dung đã làm

Thêm `CausalSelfAttention` trong `src/llm_reasoning/attention.py`. Khối nhận `[batch, sequence, d_model]`, trả về tensor cùng shape. Dùng các phép toán PyTorch trực tiếp, không dùng `nn.MultiheadAttention` hoặc hàm attention đóng gói trong implementation.

Các bước forward:

1. Chiếu Q/K/V bằng ba ma trận không bias.
2. Tách tensor thành `[batch, heads, sequence, head_dim]`.
3. Áp dụng RoPE từ task 2.1 lên Q và K, với vị trí bắt đầu từ 0.
4. Tính `Q @ K.transpose(-2, -1) / sqrt(head_dim)`.
5. Mask các vị trí tương lai bằng `-inf`, giữ đường chéo để token nhìn thấy chính nó.
6. Softmax theo chiều key, nhân V, ghép head và chiếu output không bias.

Với cấu hình mục tiêu hidden 512, 8 head, mỗi head có 64 chiều. Bốn ma trận chiếu chứa tổng `4 * 512 * 512 = 1.048.576` tham số mỗi lớp attention. Khối không chứa normalization hoặc residual; Pre-Norm và residual sẽ ghép ở task 2.3.

## Ổn định số

Điểm attention, softmax và phép nhân xác suất với V tính trong FP32 đối với đầu vào FP16/BF16/FP32. FP64 được giữ khi kiểm tra gradient số. Vùng này tắt ambient autocast để matmul không bị tự chuyển trở về half sau khi đã ép kiểu FP32. Kết quả attention được đưa về dtype của V trước output projection.

Đây là attention tường minh có bộ nhớ tăng theo bình phương chiều dài chuỗi. Chưa có kernel tối ưu; tốc độ và VRAM cần đo trên T4 ở các task sau. Không suy diễn hiệu năng train từ kiểm thử CPU.

## Kiểm thử

Thêm 9 trường hợp kiểm thử trong `tests/test_attention.py`:

- Đối chiếu output FP64 với `scaled_dot_product_attention` của PyTorch có causal mask. Hàm này chỉ dùng làm tham chiếu trong test.
- Thay đổi token tương lai và kiểm tra prefix output giữ nguyên.
- Output prefix của chuỗi dài tương đương output khi chạy riêng prefix.
- Gradient từ loss prefix về input tương lai bằng 0.
- Chuỗi một token tương đương `out_proj(v_proj(x))`; backward có gradient hữu hạn.
- Kiểm tra gradient số với FP64.
- FP16/BF16/FP32 trên input lớn có output và gradient hữu hạn.
- CPU BF16 autocast không tạo NaN/Inf trên dữ liệu thử.
- Từ chối shape, kiểu số nguyên, sequence rỗng, sequence vượt context và kích thước head không hợp lệ.

Lệnh đã chạy trên CPU, PyTorch đã cài ở task 2.1:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv\Scripts\python.exe -m pytest -q
```

Kết quả toàn bộ suite: **39/39 kiểm thử đạt trong 2,52 giây**. Đặt số thread bằng 1 cho workload kiểm thử nhỏ, không phải cấu hình hiệu năng train. Chưa kiểm thử trên GPU hoặc chạy huấn luyện chính.

## Giới hạn và bước tiếp theo

Khối hiện xử lý contiguous causal sequence, không có KV cache; task 2.5 sẽ bổ sung cache và offset tương ứng. Chưa hỗ trợ left padding hoặc mask ngăn attention giữa các tài liệu. Right padding không tác động token hợp lệ ở phía trước, nhưng training loop phải mask loss tại padding. Không dùng `loss_mask` SFT làm attention mask: mô hình vẫn cần nhìn thấy prompt câu hỏi.

Task 2.3 tiếp theo sẽ ghép embedding, các decoder block Pre-Norm, final norm và LM head; thực hiện weight tying, kiểm tra forward/backward và đếm số tham số toàn mô hình. Task 2.2 chưa cần notebook hoặc Kaggle GPU.
