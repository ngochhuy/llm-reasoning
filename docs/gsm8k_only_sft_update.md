# Chuyển SFT sang GSM8K

Theo quyết định của người thực hiện, ngừng mở rộng bộ lọc Orca. Luồng SFT chính chỉ tải và chuẩn bị GSM8K, tỷ lệ theo nguồn là GSM8K 100%, Orca 0%. Mã khảo sát/lọc Orca cũ được giữ để tham khảo lịch sử; không gọi trong pipeline chính.

## Thay đổi

- `sft_data.py`: bỏ Orca khỏi API `prepare_sft` và CLI; manifest ghi `sft_sources: ["gsm8k"]`. Từ chối thư mục còn `orca_train.jsonl` để tránh dùng nhầm dữ liệu cũ.
- Cấu hình: bỏ số lượng Orca và mức tối thiểu 5M token; bổ sung tối đa 3 epoch, giữ trần 8M token. Đây là cấu hình bàn giao cho training loop sẽ xây ở giai đoạn 2, chưa phải một run train đã thực hiện.
- Notebook 01–02: khảo sát và chuẩn bị GSM8K; đầu ra mới `data/sft_gsm8k/`; bỏ cờ `--orca-limit`. Xóa output cũ để tránh hiển thị kết quả không còn đúng với code. Khởi động lại kernel trước khi chạy bản mới.
- Đề cương, kế hoạch và README được đồng bộ. Bắt đầu 1 epoch, tối đa 3 epoch, chọn checkpoint bằng validation. Không tăng số lượt lặp chỉ để đạt một mức token tối thiểu.

## Kiểm chứng

22 kiểm thử tự động đạt, gồm kiểm tra split/mask, GSM8K-only và từ chối artifact trộn cũ. Code cells notebook 01–02 phân tích cú pháp thành công.

Đã tạo lại artifact tại `artifacts/data/sft_gsm8k/` từ cache nguồn revision `740312add88f781978c0658806c59bc2815b9866`, dùng tokenizer thử hiện có:

| Split | Số câu | Token |
| --- | ---: | ---: |
| Train sau lọc | 6.969 | 1.816.382 |
| Validation | 500 | 131.059 |
| Test giữ riêng | 1.319 | Không mã hóa thành SFT |

4 câu train bị loại vì lời giải quá ngắn. Một epoch với tokenizer thử tương ứng khoảng 1,82M token toàn chuỗi; ba epoch khoảng 5,45M. Đây không phải số token response-only hoặc thời gian train. Cần tạo lại SFT bằng tokenizer chính thức sau khi chuẩn bị dữ liệu đầy đủ; số token và số câu giữ lại có thể thay đổi. Dữ liệu ít hơn làm tăng rủi ro overfit; theo dõi validation và báo cáo giới hạn năng lực mô hình.
