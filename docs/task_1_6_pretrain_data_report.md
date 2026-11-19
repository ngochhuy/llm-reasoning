# Báo cáo task 1.6 — chuẩn bị dữ liệu pretrain

**Trạng thái:** Mã pipeline và lệnh chạy Kaggle CPU đã hoàn thành; đã kiểm tra bằng dữ liệu giả lập. Chưa chạy trên FineWeb-Edu/OpenWebMath thật, nên chưa có token, thời gian hay số shard thực tế.

## Đã triển khai

- `src/llm_reasoning/pretrain_data.py` mở hai dataset bằng Hugging Face `load_dataset(..., streaming=True)` và chốt revision SHA trước khi đọc. FineWeb-Edu dùng config `sample-10BT`; OpenWebMath dùng split `train`. Dependency `datasets` nằm trong nhóm `data` của `pyproject.toml`; `uv.lock` đã cập nhật.
- Mẫu huấn luyện tokenizer lấy từ tài liệu được phân vào **train** bằng hash ổn định, xen bốn hàng FineWeb-Edu với một hàng OpenWebMath. Validation không tham gia train tokenizer. Tokenizer được lưu trước khi mã hóa toàn bộ corpus; manifest ghi SHA-256 của file tokenizer.
- Văn bản được chuẩn hóa, loại rỗng và trùng bằng `data_prep.py`. Cùng một `seen_keys` dùng cho cả hai nguồn để loại trùng xuyên nguồn. Mỗi tài liệu thêm `<eos>`; token được đóng gói liên tục thành block 512, lưu ở file uint16 với giới hạn 2.048 block/shard mặc định. Train và validation ghi vào thư mục khác nhau.
- Ngân sách 80M/20M được tính bằng **token thật của tokenizer dự án**. Nếu tài liệu cuối vượt ngân sách, chỉ lấy phần vừa đủ và thêm EOS. Token dư chưa đủ một block được bỏ ở cuối mỗi split và ghi vào manifest.
- Manifest từng nguồn lưu revision, seed, số hàng đã xem, train/validation, số mẫu rỗng/trùng, token đã mã hóa, số token ghi thành block, danh sách shard và tail bị bỏ. Manifest tổng hợp ghi tokenizer hash, vocab thực tế và thông tin hai nguồn.

## Kiểm tra đã chạy

`uv run --offline pytest -q`: **12/12 test đạt**. Hai test mới xác nhận mẫu tokenizer gồm cả hai nguồn, ngân sách token chính xác trên dữ liệu giả lập, shard nhị phân có block cố định và chứa EOS. `git diff --check` không báo lỗi nội dung.

Lệnh chạy thử Kaggle CPU trước khi chuẩn bị đầy đủ:

```bash
uv sync --group data
uv run --group data python -m llm_reasoning.pretrain_data \
  --output /kaggle/working/llm-reasoning \
  --fineweb-tokens 100000 --math-tokens 25000
```

Sau khi xem manifest và thời gian chạy thử, bỏ hai tham số ngân sách để dùng mục tiêu 80M/20M. Script sẽ đọc lại luồng từ đầu để mã hóa bằng tokenizer vừa train; cần giữ kết nối mạng và đủ thời gian CPU.

## Giới hạn và việc cần xác nhận trên dữ liệu thật

1. **Chưa đo throughput.** BPE tự viết hiện đếm cặp theo từng vòng merge; mục tiêu 8.000 token trên mẫu 2 MB có thể tốn thời gian CPU. Cần đo thời gian train tokenizer và encode ở lần chạy thử. Nếu quá chậm, tối ưu thuật toán/caching trước khi tạo 100M token; không tự thay tokenizer sau khi đã pretrain.
2. **Lọc chất lượng OpenWebMath chưa được triển khai.** Bước này thuộc lựa chọn dữ liệu ở task 1.3 và phải được chốt trên mẫu thật trước khi chạy ngân sách đầy đủ. Hiện pipeline chỉ loại rỗng/trùng; trang lạc đề vẫn có thể đi vào corpus. Nếu mẫu không phù hợp, thay phần toán bằng FineMath theo đề cương và cập nhật pipeline/revision trước run GPU.
3. **Giới hạn độ dài và thống kê phân vị chưa chốt.** Một tài liệu rất dài có thể chiếm đáng kể ngân sách; việc cắt tài liệu cuối chỉ giữ đúng ngân sách, chưa phải bộ lọc chất lượng. Cần khảo sát mẫu ngẫu nhiên và định ra ngưỡng nếu cần.
4. **Chưa kiểm tra trùng với GSM8K test ở mức nội dung pretrain.** Cần kiểm tra trước khi dùng artifact để train. Manifest hiện không chứng minh đã loại mọi câu benchmark khỏi nguồn web.
5. Dữ liệu được lưu bằng uint16 theo byte order của máy chạy. Kaggle T4 dùng môi trường little-endian; loader ở giai đoạn huấn luyện phải kiểm tra block size, vocab hash và byte order trong manifest trước khi đọc.

**Đánh giá:** Phần cơ học streaming, token budget và ghi shard đã sẵn sàng để kiểm tra Kaggle CPU. Chưa nên dùng artifact cho run GPU chính trước khi hoàn tất kiểm tra chất lượng và hiệu năng trên nguồn thật.
