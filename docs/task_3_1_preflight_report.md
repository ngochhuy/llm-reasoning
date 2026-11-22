# Báo cáo task 3.1 — Kiểm tra trước GPU run

## Đã thực hiện

- Thêm CLI `python -m llm_reasoning.preflight`. Công cụ chỉ đọc mã/dữ liệu, kiểm tra đường output bằng file tạm rồi xóa, ghi JSON và trả mã lỗi nếu chưa đủ điều kiện. Không huấn luyện.
- Kiểm tra SHA commit, repo sạch, commit đã theo kịp `origin/main`, runtime Kaggle, CUDA, quota GPU do người dùng xác nhận, hash tokenizer giữa pretrain/SFT, format và kích thước shard, số token train, validation shard, GSM8K-only split/row count, nơi ghi và dung lượng trống tối thiểu 2 GiB.
- Notebook 04 gọi cùng preflight và chặn `RUN_MODE='production'` khi còn mục fail/pending. Chế độ `inspect` và `smoke` vẫn dùng được để xem lỗi mà không khởi động run chính.
- Thêm hai test cho trường hợp thiếu artifact và artifact dry run. Toàn suite **125 passed, 3 skipped**; notebook 04→05→06 smoke CPU vẫn chạy đạt.

## Kết quả thực tế trên máy local

| Hạng mục | Kết quả |
|---|---|
| Commit hiện tại | `95766f0b7a3c527fbfb43f7cd210c2b4e23f3bab`; nhánh `main` còn ahead `origin/main` 3 commit tại thời điểm kiểm tra. Sau thay đổi task này, working tree cần commit. |
| GPU local | NVIDIA RTX 4050 Laptop 6.141 MiB; môi trường Python `.venv` đang dùng PyTorch CPU, không đại diện Kaggle T4. |
| Tokenizer | Vocab thực tế 6.254; SHA256 `47eac6bc122ce0d17d6cccdcc1745059f9a65efe125867528500ddf58b1cec7e` khớp manifest pretrain và SFT. |
| Pretrain train | FineWeb **99.840/80.000.000** token; OpenWebMath **24.576/20.000.000** token. Đây là artifact dry run. |
| Pretrain validation | Cả hai nguồn **0 token/không có shard**. |
| SFT | Chỉ GSM8K; 6.969 train, 500 validation, 1.319 test; số dòng khớp manifest. |
| Output local | Thư mục `artifacts/preflight_output` ghi được và đủ dung lượng. Thư mục tạm dùng bởi smoke test có quota dưới 2 GiB, nên riêng phép kiểm dung lượng tại đó báo fail; smoke vẫn hoàn tất. |
| Kaggle | Chưa kiểm tra được GPU, quota, đường `/kaggle/input/...` và lưu bền vững từ máy local. |

Kết quả máy local được ghi trong `artifacts/task_3_1_preflight_local.json` (artifact được `.gitignore` loại khỏi repo). Lệnh preflight trả mã 1 như dự kiến vì chưa đạt điều kiện.

## Việc cần hoàn tất trước task 3.2

1. Push các commit hiện có và commit task 3.1; dùng SHA cuối cùng để điền `REPO_REF` trong notebook 03–06.
2. Tạo lại dữ liệu pretrain **chính thức** trên Kaggle CPU bằng notebook 02: đặt `REPO_REF` là SHA đã push, `RUN_PRETRAIN=True`, `RUN_SFT=True`, `DRY_RUN=False`; đạt gần 80M/20M train token, validation shard >0. Gắn toàn bộ artifact thành Kaggle Dataset. Không dùng artifact dry run local.
3. Kiểm tra GPU hours và giới hạn phiên trong Kaggle account; bật GPU T4, xác nhận PyTorch CUDA và dung lượng `/kaggle/working`.
4. Chạy notebook 04 ở `inspect`, đặt đúng đường input/output. Xem từng mục preflight; chỉ khi tất cả pass mới chuyển sang `production` và bắt đầu pilot 200–500 update của task 3.2.

Ví dụ lệnh sau khi đã gắn Dataset và kiểm tra quota:

```bash
python -m llm_reasoning.preflight \
  --data-root /kaggle/input/llm-reasoning-data \
  --output-root /kaggle/working/pretrain_run \
  --repo /kaggle/working/llm-reasoning-repo \
  --expected-commit <SHA_40_KY_TU> \
  --quota-confirmed \
  --report /kaggle/working/preflight.json
```

**Trạng thái:** đã triển khai và chạy kiểm tra task 3.1; **chưa đạt điều kiện bắt đầu GPU run**. Chưa huấn luyện mô hình chính, chưa thực hiện task 3.2.

## Cập nhật sau khi kiểm tra ZIP Kaggle

`results.zip` tải từ notebook 02 chứa đầy đủ pretrain: 80.000.000 token FineWeb, 19.999.744 token OpenWebMath, hai tập validation và tokenizer BPE 8.000. File trong ZIP qua kiểm tra CRC, kích thước shard và hash tokenizer. Artifact SFT chính thức chưa có; chỉ có dữ liệu smoke. Theo quyết định mới, notebook 04 dùng preflight `require_sft=False` (CLI `--pretrain-only`) và tự tìm `pretrain/manifest.json` trong Dataset đã gắn ở `/kaggle/input`. Sau pretrain, tạo SFT từ cùng tokenizer trước notebook 06. Các bước 2 và 4 trong danh sách cũ ở trên được thay bằng quy trình này; vẫn phải kiểm tra GPU, quota, commit và output trước khi bật `production`.
