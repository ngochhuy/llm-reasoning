# Báo cáo task 1.7 — dữ liệu math SFT

**Trạng thái:** Đã triển khai và kiểm tra pipeline trên dữ liệu giả lập. Chưa tải/chạy toàn bộ Orca Math và GSM8K trên Kaggle CPU; chưa có số mẫu, tỷ lệ loại hay số token thực tế.

## Nội dung đã làm

`src/llm_reasoning/sft_data.py` dùng tokenizer đã được train ở task 1.6. Script chốt revision SHA của Orca Math và GSM8K, tải GSM8K `main` train/test cùng Orca Math train, sau đó tạo:

| Artifact | Nội dung |
|---|---|
| `orca_train.jsonl` | Tối đa 40.000 câu Orca Math đã qua bộ lọc, sắp theo hash câu hỏi và seed. |
| `gsm_train.jsonl` | GSM8K train sau khi dành 500 câu validation. |
| `gsm_validation.jsonl` | 500 câu hỏi được tách cố định từ GSM8K train, trừ mẫu lỗi/quá dài. |
| `gsm_test.jsonl` | Bản test gốc để đánh giá về sau; không tạo `input_ids` hay response mask. |
| `manifest.json` | Revision, seed, tokenizer hash, số mẫu đã xem/giữ/loại, tổng token và phân vị độ dài theo split. |

GSM8K lấy lời giải trước `####`, bỏ marker phép tính `<<...>>` nhưng giữ số hiển thị trong lời giải; đáp số sau `####` được chuẩn hóa dấu phẩy. Orca Math chưa có trường đáp số riêng nên dùng quy tắc thận trọng: yêu cầu lời giải ít nhất 20 từ, có ít nhất hai số và số cuối nằm gần cuối câu trả lời. Câu trùng với bất kỳ câu GSM8K train/validation/test nào theo hash câu hỏi chuẩn hóa sẽ bị loại khỏi Orca.

Mẫu SFT có dạng `Question: ... <think>lời giải</think><answer>đáp số</answer><eos>`. `loss_mask` bằng 0 trên phần prompt câu hỏi và bằng 1 trên phần lời giải/đáp số/EOS; chiều dài mask bằng chiều dài `input_ids`. Mẫu dài quá `max_seq_len=1024` bị loại toàn bộ. Không cắt giữa lời giải hoặc bỏ mất đáp số. Việc dịch mask một token cho causal LM loss thuộc task training loop 2.6.

## Kiểm tra

`uv run --offline pytest -q`: **14/14 test đạt**. Test SFT xác nhận chuyển đổi `<<...>>`/`####`, tách train/validation, loại trùng Orca với câu test, mask prompt, giữ EOS và loại mẫu quá dài thay vì truncate. Lệnh CLI có thể gọi bằng `python -m llm_reasoning.sft_data`; cách chạy Kaggle CPU đã ghi trong README.

## Điểm cần review trước GPU train

1. **Độ đúng của đáp số Orca:** heuristic “số cuối” có thể nhầm số phụ ở cuối lời giải. Phải kiểm tra ngẫu nhiên ít nhất 100 mẫu Orca được giữ, ghi tỷ lệ đáp số sai; nếu tỷ lệ đáng kể, siết quy tắc hoặc chỉ dùng các mẫu có câu kết luận rõ ràng. Không đánh giá độ đúng bằng GSM8K test.
2. **Số mẫu giữ lại:** mục tiêu là 20–40 nghìn Orca. Pipeline quét các mẫu đã làm sạch đến khi nhận đủ 40 nghìn, nhưng có thể ít hơn. Manifest là nguồn sự thật; nếu dưới 20 nghìn, cần điều chỉnh bộ lọc trên validation/mẫu Orca trước run chính.
3. **Độ dài và ngân sách SFT:** cần xem p50/p95 và số mẫu >1024 trên dataset thật. 5–8 triệu token trong đề cương tính theo số token **đã xử lý khi train**, gồm các epoch nếu lặp; artifact chỉ báo số token của một lượt dữ liệu, chưa quyết định số epoch.
4. **Rò rỉ gần trùng:** hiện mới so khớp hash câu hỏi sau chuẩn hóa Unicode/khoảng trắng/chữ hoa. Các câu paraphrase hoặc đổi số chưa được phát hiện. Cần kiểm tra thêm mẫu nghi vấn trước khi dùng dữ liệu SFT.
5. **Nhật ký chất lượng:** manifest báo số mẫu không trích được đáp số, quá ngắn, quá dài và số mẫu giữ lại. Nếu tỷ lệ loại cao, xem ví dụ lỗi thay vì nới bộ lọc chỉ để đạt đủ số lượng.

**Kết luận cho review:** pipeline tạo artifact SFT với mask và ranh giới test rõ ràng. Chất lượng nhãn Orca và độ dài token phải được kiểm chứng bằng lần chạy Kaggle CPU trước giai đoạn huấn luyện.
