# Báo cáo task 1.8 — notebook khảo sát và chuẩn bị dữ liệu

**Trạng thái:** Đã tạo hai notebook của giai đoạn 1 và kiểm tra cấu trúc/cell smoke bằng dữ liệu giả lập. Chưa chạy các cell truy cập Hugging Face trên dữ liệu thật. Notebook có thể chạy trên máy local hoặc Kaggle CPU.

## Sản phẩm

| Notebook | Nội dung | Đầu ra để review |
|---|---|---|
| `notebooks/01_data_exploration.ipynb` | Kết nối repository, ghi commit; mở mẫu nhỏ của FineWeb-Edu, OpenWebMath, Orca Math, GSM8K train bằng streaming; in revision, schema, độ dài ký tự, mẫu và tỷ lệ sơ bộ parse đáp số. | Bảng thông tin nguồn và nhận xét quy tắc lọc trước khi chuẩn bị dữ liệu thật. |
| `notebooks/02_tokenizer_data_preparation.ipynb` | Smoke test tokenizer và pipeline bằng dữ liệu giả lập; tùy cờ chạy pretrain data và SFT data thật; đọc manifest và nhắc lưu artifact. | `tokenizer.json`, shard pretrain, SFT JSONL, manifest và các số liệu token/mẫu thực tế sau khi chạy Kaggle. |

Mã BPE và xử lý dữ liệu vẫn nằm trong module `.py`; notebook chỉ gọi lại mã đó. Cả hai notebook có ô `REPO_REF` để ghim commit, ghi phiên bản mã nguồn và có phần Markdown mô tả cách đọc kết quả. Notebook 02 dùng hai cờ `RUN_PRETRAIN`/`RUN_SFT`; giá trị mặc định là `False` để không vô tình tải và xử lý dữ liệu lớn. `DRY_RUN=True` đặt ngân sách thấp để đo CPU trước lần đầy đủ.

Khi chạy local, notebook dùng repository hiện có và ghi artifact vào `artifacts/`, không clone hoặc checkout nhánh Git. Dùng `uv sync --group data --group notebook` rồi chọn `.venv` làm kernel trong VS Code. Khi chạy Kaggle, notebook clone repository và có thể checkout `REPO_REF`. Trong lần chạy chính thức, đặt `REPO_REF` thành commit SHA để cố định mã nguồn.

## Đã kiểm tra

- Hai file là JSON notebook hợp lệ (nbformat 4); mọi code cell đều compile thành công trên Python 3.11.
- Chạy cell smoke của notebook 02 trong thư mục tạm: tokenizer train/save/load, round-trip Unicode và special token, tạo shard 512 token trong block nhỏ, tạo SFT split/mask. Kết quả: `Smoke passed`, nhận 3 ví dụ GSM8K train trong mẫu giả lập.
- Bộ test của project vẫn cần chạy khi hợp nhất thay đổi; các cell đọc dataset thật chỉ có thể xác nhận trong Kaggle CPU có Internet.

## Giới hạn và điều kiện chuyển sang GPU

1. Notebook 01 đang đọc **100 hàng đầu** mỗi nguồn. Đây là bước kiểm tra schema và ví dụ, chưa đủ để ước tính tỷ lệ lạc đề hay độ dài đại diện; trước run đầy đủ cần lấy mẫu rộng hơn hoặc kiểm tra thủ công output sau xử lý.
2. Notebook 02 dùng script pretrain hiện chưa có bộ lọc chất lượng mạnh cho OpenWebMath. Phải xem mẫu toán và báo cáo task 1.6 trước khi tạo artifact 80M/20M.
3. Cần kiểm tra thủ công ít nhất 100 đáp số Orca được giữ vì quy tắc trích số cuối có thể sai. Notebook hiện in manifest, chưa tự tạo giao diện kiểm tra ngẫu nhiên nhãn; người thực hiện có thể đọc `orca_train.jsonl` và lời giải gốc trong dataset để đối chiếu.
4. Chạy `RUN_PRETRAIN=True`, `DRY_RUN=True` trước để đo thời gian train BPE và encode. Sau đó mới dùng `DRY_RUN=False`. Chạy SFT bằng **tokenizer đã lưu từ pretrain**; nếu chuyển phiên, gắn output phiên trước và chỉnh `PRETRAIN_DIR`.
5. Trước giai đoạn 3, lưu Kaggle output bền vững, xác nhận tokenizer hash, revision dataset, số token mỗi nguồn, số Orca/GSM giữ lại, p95 độ dài và tỷ lệ mẫu bị loại. Nếu thiếu dữ liệu hoặc tốc độ CPU không đáp ứng, xử lý ở giai đoạn dữ liệu trước khi bật GPU.

**Đánh giá:** Hai notebook đã đủ cấu trúc để chạy kiểm tra và tạo artifact bằng các module hiện có. Việc nghiệm thu dữ liệu thật còn phụ thuộc lần chạy Kaggle CPU và review chất lượng dữ liệu.
