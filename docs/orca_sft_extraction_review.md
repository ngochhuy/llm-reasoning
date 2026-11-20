# Kiểm tra sửa trích xuất đáp số Orca Math

## Lỗi đã xử lý

Bộ trích xuất cũ lấy số cuối trong 180 ký tự cuối lời giải. Trong 100 mẫu thử cũ, ít nhất 21 nhãn sai hoặc không biểu diễn đủ đáp án. Ví dụ lời giải kết luận `5 ngày` nhưng nhãn là `230`, và `3.981,85` bị đọc thành `3`.

Bộ trích xuất mới chỉ chấp nhận lời giải có đúng một giá trị số ở câu kết luận cuối cùng. Số có dấu phân cách hàng nghìn và phần thập phân được đọc nguyên vẹn; kết luận nhiều số, tỷ lệ hoặc biểu thức chứa π bị loại để tránh gán nhãn sai. Khi áp lại lên 100 mẫu cũ: 66 được giữ, 34 bị loại; nhãn `42` của `42,000` được sửa thành `42000`.

## Kết quả tạo lại tập thử

Dữ liệu nguồn lấy từ cache cục bộ, cùng revision và tokenizer như lần thử trước. Artifact kiểm tra nằm ở `artifacts/data/sft_fixed_dry_run/`, không ghi đè `artifacts/data/sft/`.

| Split | Đã xét | Giữ | Loại vì đáp số mơ hồ | Quá dài |
| --- | ---: | ---: | ---: | ---: |
| Orca train | 155 | 100 | 51 | 4 |
| GSM8K train | 6.973 | 6.969 | 0 | 0 |
| GSM8K validation | 500 | 500 | 0 | 0 |

Các hàng JSONL của tập mới đã được kiểm tra độ dài `input_ids`/`loss_mask`, giới hạn vocab và response mask; không có lỗi cấu trúc. Toàn bộ 21 kiểm thử tự động đều đạt.

## Giới hạn chất lượng nguồn

Quy tắc số duy nhất không xác minh phép toán hoặc sự phù hợp giữa câu hỏi và lời giải. Khi rà 100 mẫu mới vẫn thấy ít nhất bốn trường hợp cần loại thủ công: câu 003 kết luận “ít hơn” khi đề hỏi “nhiều hơn”; câu 008 hỏi chi phí nhưng giải ra cạnh hình lập phương; câu 028 đưa ngưỡng `18 phút` thay vì phần bể được hỏi; câu 090 có đáp án không xác định duy nhất nhưng lời giải chọn `-3`. Các chỉ số `003`, `008`, `028`, `090` là vị trí hàng bắt đầu từ 0 trong file Orca thử mới.

Vì vậy tập thử mới chỉ xác nhận bộ trích xuất hoạt động tốt hơn, chưa chứng minh toàn bộ Orca Math đủ sạch để huấn luyện. Trước khi tạo SFT chính thức, cần quyết định mức sử dụng Orca hoặc thêm quy trình duyệt và loại các mẫu sai về nội dung. GSM8K có đáp án `####` rõ ràng nên là nguồn SFT đáng tin cậy hơn cho lần train chính.
