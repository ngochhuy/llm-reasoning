# Báo cáo task 2.4 — Sinh token tự hồi quy

## Kết quả

Đã bổ sung `src/llm_reasoning/generation.py`, gồm `sample_next_token` và `generate`. Đây là decoding bằng PyTorch, dùng Transformer của task 2.3; chưa có KV cache. Trọng số kiểm tra là ngẫu nhiên, không thể dùng nội dung sinh ra để kết luận năng lực ngôn ngữ/toán học.

### Chọn token

- `temperature=0`: greedy argmax; nếu hòa, chọn ID đầu tiên theo quy tắc PyTorch. Không áp dụng top-p trong nhánh này.
- `temperature>0`: chia logits theo temperature rồi softmax; top-p giữ tập token theo xác suất giảm dần, bao gồm token đầu tiên làm xác suất tích lũy vượt ngưỡng.
- `top_p=1`: lấy mẫu trên toàn vocab. Luôn giữ ít nhất một token khi top-p rất nhỏ.
- Hỗ trợ `torch.Generator` riêng để tái lập sampling; generator phải cùng device với logits. Cùng seed không bảo đảm kết quả giống nhau giữa mọi GPU/CPU hoặc phiên bản PyTorch.
- Logits phải hữu hạn; NaN/Inf gây lỗi rõ ràng. Tính xác suất trong FP32, giữ FP64 nếu input FP64 hoặc temperature quá nhỏ để biểu diễn bằng FP32. Trừ max trước khi chia temperature để giảm rủi ro overflow.

### Vòng lặp sinh

`generate` nhận một prompt không padding `[1, T]` và trả tensor **gồm prompt cùng token mới**. Mỗi bước chạy lại toàn bộ prefix, lấy logits vị trí cuối và nối token tiếp theo. Không sửa tensor prompt đầu vào.

Dừng khi đạt `max_new_tokens`, chạm context tối đa, hoặc sinh EOS mới. EOS được giữ trong output; EOS có sẵn trong prompt không tự động kết thúc lượt sinh. Nếu hết chỗ trong context, trả về ngay; không cắt prompt hoặc đặt lại vị trí RoPE. `max_new_tokens=0` cũng trả prompt mà không chạy forward.

Chạy trong inference mode, tạm chuyển model sang eval và khôi phục trạng thái train/eval của từng module sau khi xong hoặc gặp lỗi. Không tự chuyển model sang CPU/GPU: prompt phải ở cùng device với embedding.

## Kiểm thử

Thêm 14 trường hợp trong `tests/test_generation.py` (tính cả các tham số hóa):

- Greedy lấy argmax, xử lý hòa.
- Top-p giữ token vượt ngưỡng và loại đuôi phân bố; top-p nhỏ vẫn giữ token đầu tiên.
- Temperature thay đổi phân bố; sampling tái lập theo seed.
- Greedy trên Transformer thực khớp vòng lặp đối chiếu độc lập.
- EOS dừng ngay tại bước đúng, không có forward thừa.
- Giới hạn token, context, prompt không thay đổi; không tạo gradient.
- Kết nối tokenizer, sampling trên Transformer random và khôi phục trạng thái train/eval hỗn hợp.
- Khôi phục trạng thái model khi forward thất bại.
- Từ chối prompt/budget/sampling/EOS không hợp lệ; logits NaN được báo lỗi.
- Ổn định với logits FP16 lớn và temperature rất nhỏ.

Lệnh trên CPU:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv\Scripts\python.exe -m pytest -q
```

Kết quả cuối: **61/61 kiểm thử đạt trong 3,44 giây**. Chưa đo throughput hoặc VRAM trên T4.

## Ví dụ sử dụng

```python
import torch
from llm_reasoning.config import make_config
from llm_reasoning.model import TransformerLM
from llm_reasoning.tokenizer import ByteBPETokenizer
from llm_reasoning.generation import generate

tokenizer = ByteBPETokenizer.load('artifacts/data/pretrain/tokenizer.json')
model = TransformerLM(make_config('smoke').model, vocab_size=tokenizer.vocab_size)
prompt = torch.tensor([tokenizer.encode('Question: 2 + 3 = ?\n<think>\n', allowed_special=True)])
result = generate(model, prompt, max_new_tokens=32,
                  eos_token_id=tokenizer.special_to_id['<eos>'],
                  temperature=0.7, top_p=0.95,
                  generator=torch.Generator().manual_seed(42))
print(tokenizer.decode(result[0, prompt.shape[1]:].tolist()))
```

Model trong ví dụ khởi tạo ngẫu nhiên, chỉ phục vụ kiểm tra luồng sinh. Muốn trả lời có ý nghĩa cần checkpoint đã train.

## Giới hạn và bước tiếp theo

Chỉ hỗ trợ một prompt không padding mỗi lần. Có thể gọi nhiều lần để thu K ứng viên cho thí nghiệm reasoning sau này; chưa tối ưu batch inference. Không dùng HF generate hoặc inference engine đóng gói.

Task 2.5 sẽ thêm KV cache và đối chiếu logits/greedy giữa cache ON/OFF. Task 2.8–2.10 sẽ bổ sung demo nghiệm thu và notebook; task 2.4 chưa cần GPU hoặc train chính.
