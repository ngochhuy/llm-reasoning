# Báo cáo task 2.3 — Ghép Transformer decoder-only

## Kết quả

Đã tạo `src/llm_reasoning/model.py` gồm `DecoderBlock` và `TransformerLM`, dùng các khối RMSNorm, RoPE, FFN GELU và causal attention tự viết ở task 2.1–2.2.

Luồng dữ liệu:

```text
input_ids [B, T]
  → token embedding [B, T, 512]
  → 10 decoder blocks:
      x = x + attention(RMSNorm(x))
      x = x + FFN_GELU(RMSNorm(x))
  → final RMSNorm
  → LM head (weight tied với embedding)
  → logits [B, T, vocab_size]
```

Mỗi attention có 8 head, mỗi head 64 chiều. RoPE nằm trong attention, không có learned positional embedding. FFN dùng chiều trung gian 2.048. Các projection và LM head không có bias. Context tối đa theo cấu hình là 2.304 token.

`forward(input_ids)` trả logits chưa qua softmax, không tính loss. Task 2.6 sẽ thực hiện shift target, response-only mask SFT, padding mask cho loss và training loop. Số vocab là đối số bắt buộc lấy từ tokenizer thực tế; không mặc định rằng tokenizer luôn đạt 8.000 token.

## Khởi tạo và weight tying

Trọng số embedding và các linear được khởi tạo ngẫu nhiên phân phối chuẩn, độ lệch chuẩn 0,02. Output projection của attention và down projection FFN dùng độ lệch chuẩn `0,02 / sqrt(2 * n_layers)` để hạn chế tăng phương sai qua các nhánh residual. RMSNorm giữ trọng số khởi tạo bằng 1.

Khi bật `tie_embeddings`, LM head và embedding tham chiếu cùng một `nn.Parameter`, không chỉ có giá trị giống nhau. Trọng số chung chỉ được khởi tạo một lần. `num_parameters()` đếm tham số duy nhất qua `model.parameters()`.

## Số tham số

Với 10 lớp, hidden 512, FFN 2.048, vocab 8.000 và weight tying:

| Thành phần | Tham số |
| --- | ---: |
| Embedding và LM head dùng chung | 4.096.000 |
| Attention 10 lớp | 10.485.760 |
| FFN 10 lớp | 20.971.520 |
| Hai RMSNorm mỗi block, 10 block | 10.240 |
| Final RMSNorm | 512 |
| **Tổng đo bằng mã nguồn** | **35.564.032** |

Phù hợp mô tả “khoảng 36M”. Nếu dùng vocab thử 6.254 thì công thức cho 34.670.080 tham số; cần ghi số thực tế theo tokenizer chính thức trước train.

## Kiểm thử đã thực hiện

Thêm `tests/test_model.py`, gồm 8 trường hợp kiểm thử khi tính cả hai cấu hình tied/untied:

- Ghép đúng hai nhánh residual Pre-Norm.
- Shape logits đúng; causal LM loss có shift target hữu hạn; gradient hữu hạn cho toàn bộ tham số.
- Tied embedding dùng chung đối tượng tham số, untied dùng hai đối tượng riêng; số tham số khớp công thức.
- Logits quá khứ không đổi khi thay token tương lai, và khớp khi chạy riêng prefix.
- Khởi tạo tái lập khi đặt cùng seed.
- Nạp lại state dict giữ logits và weight tying. Đây là kiểm tra trọng số cơ bản; checkpoint optimizer/RNG/resume đầy đủ thuộc task 2.7.
- Cấu hình mục tiêu 10 lớp/512/8 head/2.048/vocab 8.000 chạy forward/backward trên CPU với batch 1, chuỗi 8 token. Đếm được 35.564.032 tham số.
- CPU BF16 autocast có logits và gradient hữu hạn.
- Kiểm tra biên context, shape/dtype/vocab không hợp lệ.

Lệnh:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv\Scripts\python.exe -m pytest -q
```

Kết quả: **47/47 kiểm thử đạt trong 2,92 giây** trên CPU. Số thread này chỉ để chạy kiểm thử nhỏ. Chưa đo train sequence 512, VRAM hoặc throughput trên T4. Chưa huấn luyện trên dữ liệu dự án và chưa kết luận chất lượng sinh văn bản.

## Cách sử dụng

```python
import torch
from llm_reasoning.config import make_config
from llm_reasoning.model import TransformerLM

config = make_config('smoke')
model = TransformerLM(config.model, vocab_size=config.tokenizer.vocab_size)
# Trước run chính: dùng tokenizer.vocab_size của tokenizer đã lưu.
input_ids = torch.randint(0, model.vocab_size, (2, 16))
logits = model(input_ids)
print(logits.shape, model.num_parameters())
```

## Bước tiếp theo

Task 2.4: generation bằng greedy và temperature/top-p, xử lý EOS và giới hạn token trên trọng số thử. KV cache thuộc task 2.5; notebook 03 thuộc task 2.10. Hiện chưa cần bật GPU Kaggle hoặc chạy train chính.
