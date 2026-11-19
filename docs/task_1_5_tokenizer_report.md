# Báo cáo task 1.5 — Byte-level BPE tokenizer

**Trạng thái:** Đã triển khai và kiểm tra trên dữ liệu nhỏ. Chưa huấn luyện tokenizer 8.000 token bằng mẫu dữ liệu thật; việc đó thuộc task 1.6–1.7.

## Nội dung đã làm

Triển khai `ByteBPETokenizer` trong `src/llm_reasoning/tokenizer.py` bằng Python chuẩn, không gọi thư viện tokenizer. Tokenizer có `train(texts)`, `encode(text)`, `decode(ids)`, `save(path)` và `load(path)`. Cấu hình lấy từ `TokenizerConfig`, mặc định mục tiêu 8.000 token.

Vốn từ bắt đầu bằng **256 byte UTF-8** và sáu special token (`<eos>`, `<pad>`, `<think>`, `</think>`, `<answer>`, `</answer>`). Trong lúc train, văn bản được tách thành các mảnh chữ, số, khoảng trắng và dấu; cặp token liền nhau có tần suất cao nhất trong các mảnh được gộp tuần tự. Hòa tần suất được giải quyết theo ID cặp để kết quả tái lập. Token mới được cấp ID sau nhóm byte và special token, tối đa đến `vocab_size`.

Khi encode, special token **chỉ được nhận diện nếu gọi `allowed_special=True`**. Điều này giúp một chuỗi như `<think>` nằm trong văn bản nguồn được giữ như nội dung thông thường, còn mẫu SFT có thể chèn token định dạng thực sự. Encode luôn có đường dự phòng bằng byte nên có thể xử lý từ/symbol chưa xuất hiện lúc train. File JSON lưu phiên bản định dạng, vốn từ mục tiêu, danh sách special token và thứ tự merge; load dựng lại các byte của token từ merge rules.

## Kết quả kiểm tra

Chạy `uv run --offline pytest -q`: **10/10 test đạt** cho toàn project, gồm ba kiểm tra tokenizer mới:

| Tình huống | Kết quả |
|---|---|
| Round-trip văn bản lạ có tiếng Việt, emoji, số âm, dấu phẩy/thập phân, `x²`, xuống dòng | `decode(encode(text)) == text` |
| Special token trong mẫu SFT và chuỗi giống special token trong văn bản thường | Chỉ mẫu gọi `allowed_special=True` tạo special ID; decode giữ đúng chuỗi |
| Save/load | ID encode trước và sau load giống nhau |
| Train hai lần cùng corpus | Merge rules giống nhau; văn bản lặp được mã hóa ngắn hơn chuỗi byte ban đầu |
| Đầu vào rỗng, ID lạ, phiên bản file sai | Báo lỗi rõ ràng |

Tokenizer không tự chuẩn hóa nội dung lúc encode. Văn bản train được chuẩn hóa ở `data_prep.py`; việc tách trách nhiệm này giữ nguyên tính round-trip của đầu vào encode.

## Giới hạn và việc cần làm tiếp

- Hàm train hiện giới hạn mẫu ở **2 MB UTF-8 theo mặc định** để tránh chiếm quá nhiều RAM. Mục tiêu 8.000 token có thể cần mẫu lớn hơn và thời gian gộp cặp đáng kể vì bản triển khai ưu tiên rõ thuật toán hơn tối ưu tốc độ. Task 1.6 sẽ đo thời gian/memory trên mẫu thật và quyết định kích thước mẫu train tokenizer trước khi chuẩn bị toàn bộ dữ liệu.
- `vocab_size` là **mức trần**, không bảo đảm đủ đúng 8.000 token: train dừng nếu không còn cặp xuất hiện ít nhất hai lần. Báo cáo số token thực tế sau train.
- Chưa có phép đo độ dài câu hỏi/lời giải GSM8K và Orca Math dưới tokenizer này. Task 1.7 phải đo phân bố độ dài và tỷ lệ mẫu vượt 1.024 token trước khi đóng gói SFT.
- Chưa benchmark tốc độ encode trên corpus lớn. Nếu quá chậm, có thể tối ưu bộ đếm cặp hoặc cache kết quả encode mà vẫn giữ format và ID; mọi thay đổi phải kiểm tra lại tính tương thích save/load.

## Cách kiểm tra nhanh

```python
from llm_reasoning.config import TokenizerConfig
from llm_reasoning.tokenizer import ByteBPETokenizer

tokenizer = ByteBPETokenizer(TokenizerConfig(vocab_size=512))
tokenizer.train(["Solve 12 + 30 = 42."] * 100)
ids = tokenizer.encode("<answer>42</answer>", allowed_special=True)
assert tokenizer.decode(ids) == "<answer>42</answer>"
tokenizer.save("artifacts/tokenizer.json")
```

**Kết luận cho review:** API và định dạng special token đã sẵn sàng để task 1.6 dùng. Việc chốt tokenizer 8.000 token cho lần train chính còn phụ thuộc benchmark trên mẫu dữ liệu đại diện.
