# Kế hoạch thực hiện LLM-Reasoning

## 1. Mục tiêu và cách review

Triển khai theo `De_cuong_ky_thuat_llm-reasoning_v3.md`: một mô hình khoảng 36M tham số, khởi tạo ngẫu nhiên, pretrain rồi SFT nối tiếp trên một Kaggle T4. Lịch mục tiêu là 12 ngày, gồm thời gian review.

Không tạo bản đề cương kỹ thuật 2 trang theo quyết định của người thực hiện. Bản v3 tiếp tục là tài liệu tham chiếu kỹ thuật. Tài liệu này quy định thứ tự công việc và các sản phẩm bàn giao.

- Codex triển khai từng task hoặc nhóm task phụ thuộc trực tiếp, kiểm tra local, chuẩn bị notebook và phân tích kết quả.
- Người thực hiện review, quản lý commit/push, chạy notebook trên Kaggle và cung cấp output/checkpoint.
- Sau mỗi giai đoạn, bàn giao thay đổi, bằng chứng kiểm tra, vấn đề còn lại và danh sách cần review. Chỉ sang giai đoạn tiếp theo khi người thực hiện phản hồi.
- GitHub remote đã được liên kết. Notebook Kaggle ghi commit mã nguồn đã dùng; không lấy nguồn thay đổi giữa một run.
- Pilot thuộc cùng run chính; resume tiếp tục cùng mô hình. Không khởi tạo lại một đợt train khác nếu chưa thống nhất.

## 2. Giai đoạn 1 — Nền tảng, dữ liệu và tokenizer

**Lịch mục tiêu:** ngày 1–2. **Review 1:** môi trường, dữ liệu và tokenizer.

| ID | Task | Đầu ra / điều kiện hoàn thành |
|---|---|---|
| 1.1 | Khởi tạo package Python 3.11, uv, pytest, cấu trúc artifact và gitignore | Import package được; có pyproject và lockfile; bỏ qua dữ liệu lớn, checkpoint và thông tin xác thực. |
| 1.2 | Tạo cấu hình dữ liệu, tokenizer, mô hình, train, evaluation và đường dẫn local/Kaggle | Có cấu hình kiểm tra nhỏ và cấu hình 36M; lỗi cấu hình được phát hiện sớm. |
| 1.3 | Kiểm tra mẫu FineWeb-Edu, OpenWebMath, Orca Math và GSM8K | Bảng trường dữ liệu, độ dài, chất lượng, ví dụ và quy tắc lọc. Nếu đổi OpenWebMath sang FineMath phải chốt trước run chính. |
| 1.4 | Chuẩn hóa, loại trùng và tạo split cố định | Manifest nguồn, revision, seed và số mẫu bị loại; pretrain validation 1% tài liệu, GSM8K validation 500 câu lấy từ train; không dùng test để chọn cấu hình. |
| 1.5 | Tự viết byte-level BPE khoảng 8.000 token | Có train/encode/decode/save/load, special token; round-trip văn bản, số và ký hiệu toán; save/load giữ nguyên ID. |
| 1.6 | Xây pipeline pretrain chạy trên Kaggle CPU | Tokenizer chỉ học trên mẫu train; shard có EOS, block 512; nhắm tới 80M token FineWeb-Edu + 20M token toán; ghi token thực tế và hash tokenizer. |
| 1.7 | Xây pipeline SFT | Chọn khoảng 20–40 nghìn bài Orca Math và GSM8K train; thống nhất think/answer; có response mask, EOS, thống kê lỗi và độ dài; không cắt mất đáp số. |
| 1.8 | Hoàn thiện notebook 01–02 và bàn giao | Notebook 01 khảo sát dữ liệu; notebook 02 huấn luyện tokenizer và tạo artifact. Mẫu nhỏ đã chạy, test tokenizer đạt; hướng dẫn tạo artifact đầy đủ trên Kaggle CPU và bảng cần người thực hiện review. |

**Điểm dừng:** review dữ liệu, cách chia validation, tokenizer và định dạng SFT. Dữ liệu đầy đủ phải được tạo xong trước giai đoạn chạy GPU chính.

## 3. Giai đoạn 2 — Lõi mô hình, pipeline train và nghiệm thu sơ bộ

**Lịch mục tiêu:** ngày 3–4. **Review 2:** demo lõi PyTorch và tính đúng của KV cache, trước lần train chính. Báo cáo profiling chính thức dùng checkpoint sau pretrain ở giai đoạn 3.

| ID | Task | Đầu ra / điều kiện hoàn thành |
|---|---|---|
| 2.1 | Viết RMSNorm, RoPE và FFN GELU | Pre-Norm, phép chiếu không bias; kiểm tra shape, gradient và offset RoPE. |
| 2.2 | Tự viết causal attention bằng phép toán PyTorch | Q/K/V, mask, softmax và output projection; token tương lai không ảnh hưởng output quá khứ; không NaN. |
| 2.3 | Ghép Transformer | 10 lớp, hidden 512, 8 head, FFN 2048, weight tying; forward/backward đạt; báo cáo số tham số thực tế. |
| 2.4 | Viết generation | Greedy, temperature/top-p sampling, EOS và giới hạn token; kiểm tra được trên mô hình random. |
| 2.5 | Viết KV cache | Prefill, append K/V từng lớp, reset, RoPE offset; logits gần tương đương cache OFF và greedy giống nhau trên tập kiểm tra. |
| 2.6 | Viết training loop và logging | Causal LM loss, response-only SFT, FP16/GradScaler, AdamW, accumulation, clipping, warmup/decay; overfit một batch; log TensorBoard và log dạng bảng. |
| 2.7 | Viết checkpoint/resume | Lưu model, optimizer, scheduler, scaler, RNG, vị trí dữ liệu, bước/token, cấu hình và tokenizer hash; so sánh run liên tục với run ngắt/resume. |
| 2.8 | Đóng gói demo nghiệm thu lõi | Một lệnh nạp checkpoint kiểm tra, chạy forward và sinh tự hồi quy; ghi rõ đây là checkpoint kiểm tra, chưa phải mô hình đã train chính thức. |
| 2.9 | Chuẩn bị script profiling | Script cache ON/OFF cho prompt 128/512/1024/2048; kiểm tra chạy bằng trọng số thử. Chưa lấy đây làm số liệu bàn giao chính thức. |
| 2.10 | Hoàn thiện notebook 03 và chuẩn bị 04–06 | Notebook 03 kiểm tra mô hình; notebook 04 pretrain, 05 profiling, 06 SFT gọi module đã kiểm tra bằng cấu hình nhỏ. Hướng dẫn đúng commit, dependencies, dữ liệu, checkpoint và resume; tổng hợp test để review trước run chính. |

**Điểm dừng:** chỉ chạy train chính sau review. Lỗi causal mask, cache hoặc resume phải được sửa trước. Local RTX 4050 dùng cấu hình nhỏ; profiling bàn giao dùng T4, không suy diễn tốc độ từ local.

## 4. Giai đoạn 3 — Huấn luyện một mô hình trên Kaggle

**Lịch mục tiêu:** ngày 5–7. **Review 3:** checkpoint cùng TensorBoard logs của pretrain/SFT.

| ID | Task | Đầu ra / điều kiện hoàn thành |
|---|---|---|
| 3.1 | Kiểm tra trước run | GPU, quota, commit, dữ liệu, tokenizer hash và nơi xuất artifact đúng; chuẩn bị dữ liệu hoàn tất trước khi bật GPU. |
| 3.2 | Pilot trong run chính bằng notebook 04 | 200–500 bước đầu đo throughput sau warmup, VRAM và loss; tính ngân sách thời gian còn lại rồi tiếp tục từ cùng trạng thái. |
| 3.3 | Pretrain bằng notebook 04, profiling bằng notebook 05 | Chạy đến 100M token hoặc mốc dừng theo thời gian; lưu checkpoint pretrain và loss. Notebook 05 nạp checkpoint này để demo nạp trọng số/sinh tự hồi quy và đo cache ON/OFF trên T4. Prompt 128/512/1024/2048, sinh cố định 128 token, warmup và 5 lượt đo; xuất trung vị latency, throughput, peak VRAM, bảng, biểu đồ và báo cáo ngắn. |
| 3.4 | SFT bằng notebook 06 | Nạp artifact checkpoint pretrain với optimizer/schedule SFT riêng; response-only loss; tối đa 5–8M token đã xử lý; chọn checkpoint bằng validation. Chạy trong phiên mới không yêu cầu chạy lại notebook 04. |
| 3.5 | Xuất và kiểm tra artifact | Checkpoint cuối, tokenizer, cấu hình, log dạng bảng, TensorBoard event files và biểu đồ loss; kiểm tra nạp lại từ bản đã xuất. |
| 3.6 | Phân tích và bàn giao | Báo ổn định, overfit, định dạng đầu ra, token, throughput, VRAM và thời gian GPU; chốt checkpoint đánh giá duy nhất. |

**Mặc định khởi điểm:** pretrain sequence 512; SFT tối đa 1024; AdamW learning rate pretrain 3e-4 và SFT 3e-5; gradient clipping 1.0; warmup 5%; effective batch mục tiêu khoảng 16.384 token. Chọn micro-batch bằng kiểm tra VRAM trước run và ghi cấu hình thực tế.

Checkpoint mỗi 30 phút và khi chuyển giai đoạn. Nếu chạy một phiên 12 giờ, tính thêm thời gian profiling từ pilot và dành trước 2–3 giờ cuối cho SFT, kiểm tra và xuất kết quả; giảm token pretrain nếu cần. Nếu cần đổi phiên, xác nhận artifact đã lưu bền vững và resume cùng run. Không cam kết 100M token nếu throughput không đáp ứng.

**Điểm dừng:** review mô hình và log trước đánh giá. Đường loss chưa hội tụ tốt vẫn phải báo trung thực; không tự train lại mô hình khác.

## 5. Giai đoạn 4 — Đánh giá hướng C và tổng hợp profiling

**Lịch mục tiêu:** ngày 8–10. **Review 4:** kết quả, phân tích lỗi và giới hạn kết luận.

| ID | Task | Đầu ra / điều kiện hoàn thành |
|---|---|---|
| 4.1 | Bộ trích đáp số | Xử lý answer tag, số âm, phân số, thập phân, dấu phẩy và format lỗi; test ca điển hình; báo riêng tỷ lệ không trích được. |
| 4.2 | Greedy và sampling baseline | Dự đoán từng câu validation có đáp số, đúng/sai, số token, latency, seed và trạng thái dừng. |
| 4.3 | Majority voting | Tối đa 10 ứng viên/câu; dùng tiền tố K=3/5/10; hòa phiếu chọn đáp số xuất hiện sớm nhất, toàn bộ không hợp lệ thì thất bại. |
| 4.4 | Best-of-N | Dùng cùng ứng viên; ưu tiên đáp số hợp lệ và tỷ lệ phép tính kiểm tra đúng bằng parser số học an toàn; hòa chọn ứng viên sớm nhất. Không có phép tính kiểm tra được thì chỉ xét format; báo coverage và giới hạn. |
| 4.5 | Rà báo cáo profiling từ checkpoint pretrain | Kiểm tra số liệu, biểu đồ, thông tin checkpoint và cấu hình đo của task 3.3. Không đo lại chỉ vì SFT thay đổi trọng số; nếu thay kiến trúc/code inference/cấu hình đo thì cập nhật phép đo trên cùng checkpoint pretrain. Tổng chiều dài hỗ trợ ít nhất 2304. |
| 4.6 | Hoàn thiện notebook 07, khóa cấu hình và chạy test | Notebook 07 nạp checkpoint SFT; chốt trên validation; mặc định temperature 0,7, top-p 0,95, tối đa 512 token sinh; sau khi khóa mới chạy GSM8K test. Lưu từng câu và hỗ trợ tiếp tục đánh giá. |
| 4.7 | Tổng hợp và bàn giao | Greedy accuracy, sampling Pass@1, Pass@k, voting/Best-of-N accuracy, accuracy–compute, độ dài lời giải/độ đúng; ít nhất 5 ca sai và lỗi verifier. |

Verifier không được đọc đáp án chuẩn để chọn ứng viên. Sinh một tập 10 ứng viên/câu rồi tái sử dụng cho K/N=3/5/10; báo riêng chi phí generation và selection. Khi đo GPU phải đồng bộ CUDA và reset peak memory từng lượt. Ghi rõ chế độ chạy tuần tự/batch và giữ cấu hình so sánh nhất quán.

Mục tiêu đánh giá đủ 1.319 câu GSM8K test. Nếu quota không đủ, dùng tập con cố định theo seed, công bố kích thước và không gọi đó là kết quả toàn test. Không chỉnh prompt, checkpoint hoặc verifier sau khi xem test.

## 6. Giai đoạn 5 — Báo cáo, tái lập và demo

**Lịch mục tiêu:** ngày 11–12. **Review 5:** bộ sản phẩm cuối.

| ID | Task | Đầu ra / điều kiện hoàn thành |
|---|---|---|
| 5.1 | README, notebook và lệnh tái lập | Bảy notebook gọi module dùng chung, mỗi notebook chạy được trong phiên mới bằng artifact đã lưu; có một lệnh tái lập đánh giá từ checkpoint cuối, nêu artifact/môi trường cần có. Hướng dẫn riêng train từ đầu và resume. |
| 5.2 | Báo cáo khoa học IEEE | 6–8 trang theo template IEEE conference hai cột; phương pháp, thiết lập, kết quả, phân tích lỗi, ethics và limitations; chỉ dùng số liệu thật. |
| 5.3 | Demo tương tác và slide | Demo terminal nhập câu hỏi, chọn greedy/voting/Best-of-N, hiển thị kết quả, token và thời gian; dùng checkpoint cuối. Slide có kết quả, ví dụ thành công/thất bại và giới hạn. |
| 5.4 | Rà nghiệm thu | Kiểm tra lệnh chạy, đường dẫn, khả năng tải artifact, biểu đồ, log và checklist bàn giao; ghi rõ phần chưa đạt nếu có. |

Viết nội dung phương pháp và ghi chú kết quả trong suốt quá trình; không dồn toàn bộ báo cáo sang hai ngày cuối.

## 7. Danh sách 7 notebook và luồng artifact

Notebook nằm trong `notebooks/`. Thuật toán và training loop nằm trong module `.py` dùng chung; notebook gọi module, trình bày cấu hình, kết quả và nhận xét.

| Notebook | Task liên quan | Môi trường | Đầu vào | Đầu ra lưu bền vững |
|---|---|---|---|---|
| `01_data_exploration.ipynb` | 1.3–1.4 | Local/Kaggle CPU | Mẫu bốn nguồn dữ liệu và cấu hình khảo sát | Thống kê, ví dụ, quy tắc lọc và cấu hình split. |
| `02_tokenizer_data_preparation.ipynb` | 1.4–1.8 | Kaggle CPU | Nguồn dữ liệu, quy tắc lọc/split đã chốt | Tokenizer, dữ liệu pretrain/SFT đã xử lý, split IDs, manifest và hash. |
| `03_model_sanity_checks.ipynb` | 2.1–2.8, 2.10 | Local GPU/T4 | Tokenizer, mẫu dữ liệu nhỏ và cấu hình kiểm tra | Số tham số, kết quả kiểm tra forward/backward, cache, overfit và checkpoint/resume. |
| `04_pretraining.ipynb` | 3.1–3.3 | Kaggle T4 | Tokenizer, shard pretrain, cấu hình, checkpoint resume nếu có | Checkpoint pretrain đầy đủ, TensorBoard logs, cấu hình và thống kê GPU/token. |
| `05_kv_cache_profiling.ipynb` | 2.9, 3.3, 4.5 | Kaggle T4 | Checkpoint pretrain, tokenizer và cấu hình đo | Kết quả cache ON/OFF, bảng, biểu đồ và báo cáo định lượng. |
| `06_math_sft.ipynb` | 3.4–3.6 | Kaggle T4 | Checkpoint pretrain, tokenizer, dữ liệu SFT và validation | Checkpoint SFT được chọn bằng validation, trạng thái resume và TensorBoard logs. |
| `07_reasoning_evaluation.ipynb` | 4.1–4.4, 4.6–4.7 | Kaggle T4 | Checkpoint SFT, tokenizer, validation/test và cấu hình đánh giá | Dự đoán, ứng viên, metrics, chi phí, biểu đồ và phân tích lỗi. |

Thứ tự chạy: **01 → 02 → 03 → 04 → 05 → 06 → 07**. Profiling không sửa checkpoint pretrain; notebook 06 dùng chính trọng số đã xuất từ notebook 04. Task 4.5 rà kết quả notebook 05, chỉ chạy lại phép đo khi có thay đổi cần kiểm chứng.

Mỗi notebook có bước thiết lập môi trường, kiểm tra artifact đầu vào, cấu hình, thực thi, lưu output và tóm tắt kết quả. Phải chạy được trong **phiên mới**, không dựa vào biến trong bộ nhớ hoặc chạy ngầm notebook trước. Dữ liệu/checkpoint được chuyển giữa các phiên bằng Kaggle output hoặc nguồn lưu trữ bền vững; notebook 04 và 06 có đường resume riêng. Tách notebook không tạo thêm mô hình hay buộc pretrain lại.

## 8. Đối chiếu yêu cầu bàn giao của giảng viên

| Yêu cầu | Task đáp ứng | Bằng chứng bàn giao |
|---|---|---|
| Đề cương kỹ thuật 2 trang | Bỏ theo yêu cầu người thực hiện | Không tạo sản phẩm này; v3 là tài liệu tham chiếu. |
| Demo PyTorch: forward, nạp trọng số, sinh tự hồi quy | 2.3, 2.8; xác nhận lại 3.5 | Lệnh demo, checkpoint, log/output chạy thực tế. |
| Báo cáo tốc độ và bộ nhớ cache ON/OFF | 2.9, 3.3 và 4.5 | Script chuẩn bị trước train; số liệu chính thức từ checkpoint pretrain trên T4; bảng, biểu đồ và báo cáo. |
| W&B/TensorBoard log, loss chuyên sâu | 2.6, 3.3–3.6 | TensorBoard event files và biểu đồ train/validation loss pretrain/SFT. |
| Báo cáo IEEE/ICLR 6–8 trang | 5.2 | Báo cáo IEEE conference. |
| GitHub, README, tái lập một lệnh | 5.1 | README và lệnh đánh giá từ checkpoint cuối; quy trình train từ đầu được hướng dẫn riêng. |
| Demo tương tác terminal/Gradio | 5.3 | Demo terminal dùng checkpoint cuối. |

Lịch 12 ngày là lịch thực hiện rút gọn. Hạn nộp từng cột mốc do giảng viên quy định cần được đối chiếu riêng nếu vẫn áp dụng.

## 9. Ưu tiên và task tiếp theo

- Không chạy tiếp khi causal mask/cache/resume sai, checkpoint không nạp được hoặc dữ liệu test bị dùng để huấn luyện/chọn cấu hình.
- Khi thiếu thời gian, điều chỉnh token pretrain theo pilot và giảm trang trí demo; giữ đầy đủ phép đo, phân tích lỗi và tài liệu bàn giao.
- Mỗi giai đoạn có kết quả kiểm tra hoặc log thật; mục chờ chạy Kaggle phải ghi rõ trạng thái.
- **Task triển khai đầu tiên: 1.1 — Khởi tạo project Python 3.11 với uv và khung kiểm tra.**
