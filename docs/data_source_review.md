# Khảo sát nguồn dữ liệu — task 1.3

**Ngày khảo sát:** 20/11/2026. Đây là khảo sát schema và một số hàng hiển thị trên trang dataset; chưa tải và đo thống kê trên mẫu ngẫu nhiên. Các khoảng độ dài bên dưới là giới hạn hiển thị của Data Studio, **không phải phân vị** và không được xem là thống kê của tập sẽ train. Số token cuối cùng phải đo bằng BPE của dự án ở task 1.6–1.7.

| Nguồn / vai trò | Split và trường cần dùng | Quan sát mẫu và độ dài hiển thị | Kết luận cho pipeline |
|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), pretrain tiếng Anh | `sample-10BT`, `train`; `text`, `id`; thêm `url`, `score`, `language_score` để khảo sát/lọc | Trường `text` của viewer mặc định có độ dài 150–592k ký tự. Mẫu đầu gồm bài hướng dẫn, tóm tắt khoa học và hỏi đáp sức khỏe; chất lượng và chủ đề khác nhau. `token_count` trong nguồn không thuộc BPE của dự án. | Dùng `sample-10BT` dạng stream, không tải toàn bộ tập; lấy `text`, lọc rỗng/trùng/nội dung điều hướng rõ rệt, giới hạn tài liệu quá dài trước tokenize. Đếm lại bằng BPE dự án. |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math), pretrain toán | `train`; `text`, `url`, `metadata`, `date` | 6,32 triệu hàng; `text` hiển thị 100–1,02M ký tự. Mẫu gồm một bài về Bayes, thảo luận vật lý và cả tài liệu test phần mềm. Trường `metadata` là chuỗi JSON; mẫu có `found_math` true hoặc false. | Không dùng mọi hàng mà không lọc. Ưu tiên văn bản có giải thích và biểu thức toán; dùng `found_math` làm tín hiệu hỗ trợ, không làm điều kiện duy nhất. Loại tài liệu kỹ thuật lạc đề, quá dài hoặc gần như toàn ký hiệu. |
| [Orca Math Word Problems 200K](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k), SFT | `train` 200.035 hàng; `question`, `answer` | `question` hiển thị 5–2.440 ký tự, `answer` 1–5.580. Mẫu có bài toán có lời giải nhiều bước, nhưng cũng có đáp án một câu rất ngắn. Lời giải do GPT-4 Turbo tạo, không có nhãn đáp số riêng. | Chọn 20–40 nghìn mẫu có câu hỏi rõ, lời giải có bước và đáp số trích được; kiểm tra phép tính/đáp số khi khả thi. Loại mẫu thiếu đáp án, quá dài, trùng câu hỏi hoặc chỉ đáp một cụm ngắn nếu mục tiêu là học CoT. |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k), SFT và benchmark | Config `main`; `train` 7.473, `test` 1.319; `question`, `answer` | `question` hiển thị 42–985 ký tự, `answer` 50–1.230. Mẫu lời giải chứa phép tính dạng `<<48/2=24>>` và đáp số cuối dạng `#### 72`. | Tách 500 câu validation từ `train` trước mọi xử lý; giữ `test` chỉ cho đánh giá cuối. Chuyển `####` sang `<answer>` và giữ bước giải trong `<think>`; kiểm tra đáp số trích được. |

## Ví dụ đại diện đã xem

- FineWeb-Edu: đoạn hỏi đáp/giải thích thuộc chủ đề sức khỏe và tóm tắt thiên văn; cho thấy nguồn có văn bản giáo dục nhưng không tập trung vào toán.
- OpenWebMath: bài giải thích định lý Bayes có công thức; đồng thời có tài liệu hướng dẫn test phần mềm, minh họa nhu cầu lọc chủ đề.
- Orca Math: bài hỏi người đứng thứ năm có bao nhiêu người về trước, lời giải chỉ một câu; bài khác về phép trừ nhầm có các bước tính rõ ràng.
- GSM8K: bài bán kẹp giấy dùng hai phép tính `48/2` và `48+24`, kết thúc bằng `#### 72`.

## Quy tắc lọc và chia tập

Task 1.4 đã triển khai chuẩn hóa NFC, giữ xuống dòng/ký hiệu toán, loại mẫu rỗng và trùng theo hash, chia pretrain validation theo hash tài liệu với seed, dành đúng 500 câu GSM8K train làm validation, giữ test riêng và loại câu Orca Math trùng câu hỏi GSM8K. Hàm xử lý và manifest nằm trong `src/llm_reasoning/data_prep.py`. Manifest ghi nguồn, revision, seed và số mẫu theo từng lý do; số liệu thật chỉ có sau khi chạy trên dữ liệu trong task 1.6–1.7.

1. Chuẩn hóa Unicode và khoảng trắng; giữ ký hiệu toán, dấu âm, thập phân, dấu phẩy số và LaTeX có nghĩa. Không xóa mọi dấu câu.
2. Loại hàng không có trường bắt buộc, rỗng sau chuẩn hóa, hoặc trùng chính xác theo hash nội dung. Với SFT, so khớp thêm câu hỏi đã chuẩn hóa giữa Orca Math và GSM8K train/validation/test để giảm rò rỉ benchmark.
3. Với pretrain, giới hạn độ dài tài liệu theo số ký tự trước tokenize để tránh một trang rất dài chiếm phần lớn ngân sách. Ghi số hàng và token bị loại ở mỗi bước; ngưỡng cụ thể chốt sau khi xem phân bố trên mẫu ngẫu nhiên ở task 1.6.
4. Với OpenWebMath, đo tỷ lệ chữ/số/ký hiệu và kiểm tra thủ công một mẫu sau lọc; giữ tài liệu giải thích toán. Nếu mẫu vẫn lệch mục tiêu, thay phần 20M token toán bằng FineMath như đề cương cho phép, trước run chính.
5. Với Orca Math, chỉ nhận mẫu có đáp số cuối trích được và lời giải đủ bước; kiểm tra câu hỏi/đáp số không bị cắt khi đóng gói chuỗi. Đếm tỷ lệ giữ/lọc để đánh giá có đủ 20–40 nghìn mẫu hay không.
6. Chia train/validation **trước** huấn luyện tokenizer và đóng gói. Không dùng GSM8K test để chỉnh quy tắc lọc, chọn siêu tham số hoặc verifier.

## Việc cần xác nhận bằng dữ liệu thật ở task 1.6–1.7

- Lấy mẫu ngẫu nhiên có seed từ từng nguồn, lưu revision, số hàng và ví dụ sau lọc; tính phân vị độ dài theo ký tự rồi theo BPE dự án.
- Kiểm tra metadata OpenWebMath có thể parse ổn định, tỷ lệ `found_math` và tỷ lệ trang lạc đề sau bộ lọc.
- Đo tỷ lệ Orca Math có đáp số trích được, lời giải nhiều bước và trùng với GSM8K.
- Xác nhận tỷ lệ 80M/20M theo **token sau BPE**, không theo số hàng hoặc `token_count` do nguồn cung cấp.

**Nguồn:** trang dataset chính thức của [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math), [Orca Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) và [GSM8K](https://huggingface.co/datasets/openai/gsm8k). Các nhận xét về chất lượng chỉ là suy luận từ các mẫu hiển thị và cần được kiểm tra trên mẫu ngẫu nhiên trước khi train.
