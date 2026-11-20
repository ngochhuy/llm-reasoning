# Báo cáo task 2.5 — KV cache tự xây dựng

## Nội dung hoàn thành

Thêm `src/llm_reasoning/kv_cache.py`, cập nhật `attention.py`, `model.py` và `generation.py` để hỗ trợ inference có/không có cache trên cùng trọng số.

`LayerKVCache` chứa hai tensor `[batch, heads, cached_tokens, head_dim]`: K đã xoay RoPE và V chưa xoay. `append` nối tensor mới bằng `torch.cat` theo chiều sequence. `KVCache` chứa cache riêng cho từng decoder layer, cung cấp chiều dài, tổng byte tensor và `reset()`.

Cache là trạng thái của một request, không phải tham số mô hình, không xuất hiện trong state dict. Model không giữ cache ẩn. Phải tạo mới/reset khi đổi prompt hoặc sau khi cập nhật trọng số. Cache đã dùng được gắn với model đã tạo nó; nếu đổi model, batch, head geometry, dtype hoặc device, phải reset/tạo mới.

## Prefill, decode và causal mask

1. Cache rỗng: forward toàn bộ prompt, lưu K/V từng lớp (prefill).
2. Cache có L token: chỉ forward token hoặc đoạn token mới, RoPE bắt đầu từ offset L.
3. K/V mới được nối với K/V đã lưu. Không xoay lại K cũ.
4. Causal mask so sánh vị trí tuyệt đối: chặn key có vị trí lớn hơn query. Mask có dạng `[new_tokens, cached_tokens + new_tokens]`, xử lý được cả decode một token và nối nhiều token.
5. Logits trả về chỉ ứng với input mới, không trả lại logits của prefix đã cache.

Model yêu cầu `eval()` và `no_grad()`/`inference_mode()` khi dùng cache. Nhánh training không truyền cache nên vẫn tính gradient như trước. Kiểm tra tổng prefix và đoạn mới không vượt context. Cập nhật cache được dựng tạm và chỉ commit khi toàn bộ forward thành công, tránh cache các lớp có chiều dài khác nhau nếu một lớp bị lỗi.

## Tích hợp generation

`generate(..., use_cache=True)` tạo cache mới cho mỗi lần gọi: prefill một lần, sau đó forward một token mỗi bước. `use_cache=False` giữ đường đối chiếu tính lại toàn prefix, là mặc định để tương thích task trước. EOS, giới hạn token/context và khôi phục train/eval vẫn được giữ.

Token cuối cùng được sinh không nhất thiết đã được đưa vào cache, vì lượt sinh đã dừng trước forward tiếp theo. Cache nội bộ của `generate` không được trả ra hoặc tái dùng giữa các prompt.

## Kiểm chứng

Thêm 24 trường hợp kiểm thử (gồm các tham số hóa) trong `tests/test_kv_cache.py`:

- Cache append giữ nguyên phần prefix; shape, dtype, số byte và gradient đúng.
- Đối chiếu logits toàn chuỗi với chunked prefill/decode ở 3 seed và 3 cách chia đoạn, có batch 2 và 2 decoder layer.
- Greedy cache ON/OFF giống nhau ở 3 seed; các request độc lập không rò trạng thái.
- Hook ghi chiều dài input: cache ON là `[4, 1, 1, 1, 1]`, cache OFF là `[4, 5, 6, 7, 8]` cho cùng số bước sinh.
- Reset giải phóng tham chiếu tensor và đặt offset về 0.
- Context boundary, lỗi giữa các lớp không làm thay đổi cache đã commit.
- Từ chối training, khác model, khác số lớp, batch hoặc dtype.
- Logits FP16/BF16 hữu hạn và gần tương đương full forward trong tolerance tương ứng.
- Offset ở prompt dài 128, 512, 1.024, 2.048, dùng mô hình nhỏ một lớp và context 2.304.
- EOS và giới hạn context hoạt động khi bật cache.

Lệnh:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv\Scripts\python.exe -m pytest -q
```

Kết quả: **85/85 kiểm thử đạt trên CPU trong 5,78 giây**. FP32 dùng tolerance `atol=1e-6, rtol=1e-5`; kiểm tra FP16/BF16 dùng tolerance lớn hơn vì làm tròn số.

Một phép đối chiếu bổ sung: mô hình random 2 lớp, hidden 16, batch 2, chuỗi 12 token, chia 7/3/2; sai khác logits tuyệt đối lớn nhất **5,96e-08**, cache 12 token chứa **6.144 byte tensor**. State dict không có cache. Đây là bằng chứng tính đúng trên cấu hình thử, chưa chứng minh greedy luôn giống nhau trên mọi trọng số/precision: logits sát nhau vẫn có thể đổi argmax do sai số số thực.

## Bộ nhớ và giới hạn

Dung lượng tensor K/V lý thuyết là `2 * n_layers * batch * n_heads * cached_length * head_dim * bytes_per_element`. Với cấu hình mục tiêu 10 lớp, hidden 512, batch 1, 2.048 token và FP16, K/V chiếm 40 MiB. Con số này không gồm model, activation, tensor tạm khi nối, logits hoặc CUDA allocator; không phải peak VRAM đo thực tế.

Cache động dùng `torch.cat` nên mỗi lần nối vẫn sao chép K/V cũ. Có chi phí cấp phát và tensor tạm; task này chưa khẳng định speedup. `reset()` bỏ tham chiếu tensor, không đồng nghĩa VRAM reserved của allocator giảm về 0. Tránh đổi trọng số, input prefix hoặc trạng thái precision giữa chừng khi cache chưa reset.

Chưa chạy GPU T4 hoặc đo báo cáo profiling chính thức. Task 2.9 sẽ chuẩn bị phép đo ON/OFF; báo cáo bàn giao chính thức vẫn dùng cùng checkpoint sau pretrain ở task 3.3. Không cần train mô hình để kiểm tra tính đúng KV cache như task hiện tại.

## Ví dụ và bước tiếp theo

```python
import torch
from llm_reasoning.kv_cache import KVCache
from llm_reasoning.generation import generate

# model và prompt đã có từ task 2.3–2.4.
cache = KVCache(model.config.n_layers)
model.eval()
with torch.inference_mode():
    prompt_logits = model(prompt, cache=cache)
    next_id = prompt_logits[:, -1].argmax(-1, keepdim=True)
    next_logits = model(next_id, cache=cache)
cache.reset()

output = generate(model, prompt, max_new_tokens=32,
                  eos_token_id=tokenizer.special_to_id['<eos>'], use_cache=True)
```

Task tiếp theo là **2.6: training loop và logging**, gồm causal LM loss/response-only SFT, FP16 GradScaler, AdamW, accumulation, clipping, schedule và overfit một batch. Chưa chạy train chính trong task 2.5.
