# ĐỀ CƯƠNG KỸ THUẬT — LLM-REASONING (V3)

## Xây dựng mô hình ngôn ngữ nhỏ từ đầu và đánh giá suy luận toán học tại thời điểm sinh

**Hướng:** C — Reasoning & Test-Time Compute  
**Thời gian còn lại:** 12 ngày; một sinh viên  
**Phần cứng mục tiêu:** một NVIDIA T4 16 GB trên Kaggle  
**Framework:** PyTorch; tự triển khai tokenizer, kiến trúc Transformer, attention và KV cache  
**Mô hình mục tiêu:** khoảng 36 triệu tham số, đếm và báo cáo số thực tế từ mã nguồn  
**Benchmark chính:** GSM8K  
**Mục tiêu huấn luyện:** một mô hình duy nhất, qua hai giai đoạn liên tiếp (pretrain → SFT), chọn một checkpoint cuối

**Tổ chức thực hiện:** theo `KE_HOACH_THUC_HIEN.md`, gồm 5 giai đoạn có điểm dừng review; chỉ chuyển giai đoạn sau phản hồi của người thực hiện. Không tạo bản đề cương nộp riêng 2 trang theo phạm vi đã thống nhất.

> **Phạm vi của “từ đầu”.** Tokenizer và kiến trúc được tự lập trình; trọng số mô hình khởi tạo ngẫu nhiên, không nạp checkpoint mô hình ngôn ngữ có sẵn. Có thể dùng các phép toán, bộ tối ưu và công cụ dữ liệu của PyTorch. Không dùng lớp đóng gói sẵn toàn bộ mô hình hoặc attention làm kiến trúc chính.

## 1. Mục tiêu và câu hỏi nghiên cứu

Xây dựng một decoder-only Transformer có thể huấn luyện, sinh văn bản tự hồi quy và dùng KV cache. Sau đó khảo sát việc tăng số lời giải được lấy mẫu có cải thiện độ chính xác trên GSM8K hay không, cùng chi phí thời gian và token.

- **RQ1:** Mô hình nhỏ được huấn luyện từ đầu với ngân sách GPU giới hạn đạt kết quả nào trên GSM8K?
- **RQ2:** Majority voting và Best-of-N thay đổi độ chính xác thế nào so với greedy decoding?
- **RQ3:** Khi tăng số ứng viên, độ chính xác, tổng token sinh và thời gian suy luận thay đổi thế nào?
- **RQ4:** KV cache thay đổi thời gian prefill, thời gian ra token đầu, độ trễ các token tiếp theo, throughput và VRAM thế nào ở các độ dài prompt khác nhau?

Không đặt trước mục tiêu độ chính xác cao. Kết quả không cải thiện vẫn là kết quả hợp lệ nếu quy trình đo đúng và có phân tích nguyên nhân.

## 2. Kiến trúc và ngân sách tham số

| Thành phần | Cấu hình khởi điểm |
|---|---:|
| Decoder blocks | 10 |
| Hidden dimension | 512 |
| Attention heads | 8 |
| Head dimension | 64 |
| Feed-forward dimension | 2.048 |
| Vocabulary | khoảng 8.000 token |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| FFN activation | GELU |
| Embedding và LM head | chia sẻ trọng số |
| Độ dài chuỗi pretrain | 512 token |
| Độ dài chuỗi SFT | tối đa 1.024 token, tùy phân bố độ dài và VRAM |
| Tổng chiều dài được hỗ trợ khi benchmark | ít nhất 2.304 token nếu prompt 2.048 và sinh 256 token |

Cấu hình trên có xấp xỉ **36 triệu tham số**; con số nghiệm thu là `sum(p.numel() for p in model.parameters())`. Nếu cấu hình thực tế thay đổi, cập nhật tên và bảng trước khi huấn luyện. Độ dài prompt 1.024–2.048 chủ yếu phục vụ benchmark hệ thống: mô hình pretrain trên chuỗi 512 token nên không được suy ra rằng nó hiểu tốt văn bản dài.

Các module cần tự viết: token embedding, RoPE, RMSNorm, causal multi-head attention, FFN, decoder block, LM head, generation và KV cache. Attention phải có đường forward đầy đủ để train và đường prefill/decode sử dụng cache. Tính đúng được kiểm tra trước khi chạy train chính.

## 3. Dữ liệu và tokenizer

### 3.1. Pretrain

| Nguồn | Vai trò | Ngân sách mục tiêu |
|---|---|---:|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Ngôn ngữ tiếng Anh và văn bản giáo dục | 80 triệu token |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | Tăng tỷ lệ văn bản toán | 20 triệu token |
| **Tổng** | | **100 triệu token** |

Số token được tính **sau khi mã hóa bằng tokenizer cuối cùng**, không tính theo số tài liệu hoặc số token của tokenizer có sẵn. 100 triệu là mức trần mục tiêu, không phải điều kiện bắt buộc nếu throughput thực tế không cho phép. Lưu số token đã xử lý thực tế và phân bố theo nguồn.

Với OpenWebMath, lấy mẫu và kiểm tra thủ công trước khi chốt: ưu tiên giải thích bằng ngôn ngữ tự nhiên, phép tính và bài toán có lời văn; giảm tài liệu quá dài hoặc gần như toàn LaTeX. Nếu mẫu OpenWebMath không phù hợp, thay **phần 20 triệu token** bằng một phần đã lọc của [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath). Không thay đổi nguồn sau khi bắt đầu run chính mà không ghi lại lý do.

Quy trình: lọc bản rỗng/lỗi, chuẩn hóa Unicode nhưng giữ ký hiệu toán có nghĩa, loại trùng theo hash, tách validation ở mức tài liệu, train tokenizer trên mẫu đại diện của cả hai nguồn và dữ liệu SFT, mã hóa, chèn EOS giữa tài liệu, đóng gói block. Kiểm tra trùng với GSM8K test trong phạm vi khả thi; ghi giới hạn của phép kiểm tra trong báo cáo.

### 3.2. Tokenizer tự xây

Triển khai byte-level BPE hoặc BPE có cơ chế xử lý mọi byte/ký tự để tránh token không biết. Mục tiêu khoảng 8.000 token, gồm token phân tách tài liệu và token định dạng SFT. Lưu vocabulary, merge rules và ID của special tokens. Kiểm tra `decode(encode(text))`, số âm, thập phân, dấu phẩy hàng nghìn, biểu thức toán, khoảng trắng và ký hiệu định dạng. Đếm độ dài chuỗi thực tế trên mẫu SFT trước khi chọn `max_seq_len`.

### 3.3. SFT toán học

| Nguồn | Cách dùng |
|---|---|
| [Orca Math Word Problems 200K](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | Chọn khoảng 20–40 nghìn bài phù hợp cấp độ GSM8K, có lời giải đầy đủ; lọc trùng và mẫu lỗi. |
| [GSM8K train](https://huggingface.co/datasets/openai/gsm8k) | Tách train/validation cố định; sử dụng để căn chỉnh dạng câu hỏi, lời giải và đáp số. |

Mục tiêu **5–8 triệu token đã xử lý** trong SFT, tính cả số lượt lặp. Chốt số epoch từ độ dài thực tế của dữ liệu và validation, không mặc định lặp nhiều lần GSM8K. GSM8K test chỉ dùng đánh giá cuối; không dùng để cập nhật trọng số, chọn checkpoint, chỉnh prompt, chỉnh verifier hoặc các siêu tham số.

Mọi ví dụ dùng cùng mẫu:

```text
Question: <problem>
<think>
<step-by-step solution>
</think>
<answer>
<final numeric answer>
</answer>
```

Chuẩn hóa lời giải GSM8K (bao gồm ký hiệu phép tính `<<...>>` và đáp số `####`) và Orca Math sang mẫu này. Tính loss trên phần lời giải và đáp số; mask phần prompt câu hỏi. Loại các mẫu không thể trích đáp số đáng tin cậy hoặc ghi riêng tỷ lệ lỗi chuyển đổi.

## 4. Kế hoạch huấn luyện trên Kaggle

### 4.1. Một pipeline, một mô hình cuối

1. Kiểm tra forward/backward, loss hữu hạn, gradient và khả năng overfit một batch nhỏ.
2. Kiểm tra lưu và khôi phục đầy đủ model, optimizer, scheduler, AMP scaler, bước, số token và trạng thái RNG.
3. Bắt đầu pretrain; dùng **200–500 bước đầu của chính run này** để đo token/giây và VRAM. Tiếp tục từ cùng trạng thái, không bỏ pilot rồi khởi tạo lại.
4. Dừng pretrain theo **ngân sách thời gian và số token**, lưu checkpoint.
5. Từ checkpoint đó, chuyển sang SFT với learning rate/schedule riêng; chọn checkpoint SFT theo validation.
6. Dùng một checkpoint SFT cuối cho tất cả thí nghiệm greedy, majority voting và Best-of-N. Có thể lưu checkpoint pretrain để báo cáo, nhưng không tạo nhiều mô hình ứng viên để chọn bằng GSM8K test.

Pretrain và SFT là hai giai đoạn tối ưu nối tiếp, dù nằm trong cùng một notebook hoặc lệnh chạy. Nếu phiên Kaggle phải khởi động lại, tiếp tục từ checkpoint là tiếp tục **cùng một run**, không phải huấn luyện một mô hình khác.

### 4.2. Cấu hình khởi điểm

| Mục | Quyết định |
|---|---|
| GPU | Dùng một T4; xác nhận bằng `nvidia-smi` |
| Precision | FP16 autocast và GradScaler |
| Optimizer | AdamW |
| Micro-batch | Chọn theo VRAM và throughput đo được, không cố định trước ở 1–4 |
| Effective batch | Tăng bằng gradient accumulation nếu cần |
| Ổn định | Gradient clipping, warmup, giảm learning rate |
| Validation | Theo mốc token/bước cố định, không quá dày |
| Checkpoint | Định kỳ và ngay trước khi chuyển giai đoạn; lưu ở `/kaggle/working` và xuất bản bền vững sau phiên |

Kaggle công bố giới hạn 12 giờ cho một phiên GPU và hạn mức GPU theo tuần có thể thay đổi theo tài khoản ([tài liệu](https://www.kaggle.com/docs/notebooks), [hướng dẫn hạn mức](https://www.kaggle.com/docs/efficient-gpu-usage)). Kiểm tra hạn mức hiện có trước khi lên lịch. Không giả định chọn T4×2 sẽ tự động tăng tốc nếu code chỉ dùng một GPU.

### 4.3. Dự toán và quy tắc dừng

Thời gian pretrain là `số token mục tiêu / throughput đo thực tế`. Ví dụ, 100 triệu token mất khoảng 13,9 giờ ở 2.000 token/giây, 6,9 giờ ở 4.000 token/giây, và 3,5 giờ ở 8.000 token/giây. Đây là **kịch bản để lập kế hoạch**, không phải cam kết tốc độ. Dự trù thêm thời gian SFT, validation, tải dữ liệu và lưu checkpoint.

Nếu chạy cả hai giai đoạn trong một phiên 12 giờ, dành trước **2–3 giờ cuối** cho SFT, kiểm tra checkpoint và xuất kết quả; đặt mốc dừng pretrain theo tốc độ pilot. Nếu phải dùng nhiều phiên, xác nhận checkpoint đã được lưu bền vững và có thể resume trước khi kết thúc phiên đầu. Báo cáo tổng thời gian GPU thực tế, throughput, token đã thấy, peak VRAM và số lần resume.

## 5. KV cache và benchmark hệ thống

Trước train chính, kiểm tra tính đúng của KV cache và chuẩn bị script đo. **Profiling chính thức dùng checkpoint sau pretrain trên T4**, cùng một checkpoint cho cache ON/OFF; xuất bảng, biểu đồ và báo cáo ngắn. Sau đó tiếp tục SFT từ checkpoint pretrain để đánh giá reasoning. Không yêu cầu đo lại chỉ vì SFT thay đổi trọng số; nếu sửa kiến trúc, code inference hoặc cấu hình đo thì cập nhật profiling trên cùng checkpoint pretrain.

Tự triển khai cache K/V theo từng lớp, prefill prompt, nối K/V khi decode, đặt vị trí RoPE đúng theo offset, kiểm tra giới hạn tổng chiều dài và hỗ trợ bật/tắt cache. Trước train chính, kiểm tra logits cached và non-cached gần tương đương trong FP32; sau đó kiểm tra lại FP16 với tolerance phù hợp. Ở chế độ `eval`, greedy token phải giống nhau trên một tập prompt kiểm tra.

Benchmark prompt dài **128, 512, 1.024, 2.048 token**. Mỗi lượt sinh cùng một số token mới, ví dụ 128 hoặc 256; đặt giới hạn tổng dài đủ cho prompt 2.048. Với mỗi cấu hình ghi:

- Thời gian prefill, **first-token latency** (tính từ khi nhận prompt đến token đầu), độ trễ mỗi token tiếp theo và tổng thời gian.
- Token/giây riêng cho decode và token/giây toàn yêu cầu; nêu rõ mẫu số.
- Peak VRAM; batch size, dtype, số token sinh, GPU và phiên bản phần mềm.

Warmup trước khi đo, đồng bộ CUDA quanh mốc thời gian, lặp nhiều lần và báo cáo trung vị. Bật/tắt cache phải dùng cùng checkpoint, prompt và số token sinh. Với peak memory của PyTorch, reset số liệu trước từng lượt đo ([tài liệu PyTorch](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.max_memory_allocated.html)). Nếu một cấu hình hết VRAM, ghi rõ `OOM` thay vì thay batch size âm thầm.

## 6. Thí nghiệm hướng C

### 6.1. Các phương pháp

- **Greedy:** một lời giải bằng argmax ở mỗi token.
- **Self-consistency / majority voting:** lấy mẫu tối đa 10 lời giải độc lập với nhiệt độ và `top_p` cố định, trích đáp số, chọn đáp số nhiều phiếu nhất. Báo cáo K = 3, 5, 10 bằng các tiền tố của **cùng tập ứng viên**.
- **Best-of-N:** xếp hạng cùng tập ứng viên bằng verifier không nhìn đáp án chuẩn; báo cáo N = 3, 5, 10. Không huấn luyện thêm một mô hình verifier để giữ cam kết một mô hình huấn luyện.

Verifier khả thi trong thời hạn: kiểm tra sự hiện diện và khả năng phân tích đáp số, kiểm tra các phép tính tường minh trong lời giải bằng parser số học an toàn, chấm tỷ lệ phép tính hợp lệ và quy tắc xử lý hòa điểm cố định. Không dùng `eval()` trên văn bản do mô hình sinh. Ghi tỷ lệ ứng viên parser/verifier xử lý được và ít nhất vài trường hợp verifier xếp hạng sai. Nếu không có phép tính kiểm tra được, dùng một quy tắc dự phòng đã chốt trên validation; không gọi điểm xác suất của mô hình là “xác minh tính đúng” nếu nó chỉ là heuristic.

### 6.2. Bộ trích đáp số và chỉ số

Chuẩn hóa số nguyên, số âm, dấu phẩy hàng nghìn và thập phân; quy định xử lý phân số, đơn vị và mẫu thiếu `</answer>`. Tách tỷ lệ **không trích được đáp số** khỏi tỷ lệ trả lời sai. Viết kiểm tra tự động cho các ca định dạng điển hình trước khi đánh giá.

- **Exact Match / Pass@1:** độ đúng của một câu trả lời được chọn, nêu rõ greedy hay một mẫu ngẫu nhiên.
- **Pass@k:** tỷ lệ câu hỏi có ít nhất một lời giải đúng trong k ứng viên; đây là chỉ số oracle để phân tích, không phải độ đúng của hệ thống bỏ phiếu.
- **Majority-vote accuracy** và **Best-of-N accuracy:** độ đúng của đáp số được phương pháp tương ứng chọn.
- **Chi phí:** tổng token sinh, thời gian và VRAM; biểu đồ accuracy–compute.

Đánh giá phát triển trên tập validation cố định (có thể chọn 100–200 câu để chạy nhanh), khóa prompt, seed, giới hạn token, sampling và verifier, sau đó chạy GSM8K test cuối. Không sinh lại độc lập cho mỗi giá trị K/N. Nếu ngân sách test hạn chế, ưu tiên một bộ kết quả đầy đủ cho cấu hình đã khóa; ghi rõ kích thước mẫu và không gọi tập con là toàn bộ GSM8K test. Phân tích tối thiểu năm ca sai thuộc các loại lỗi khác nhau.

## 7. Mã nguồn, notebook và tái lập

Các thuật toán nằm trong module `.py`; notebook chỉ gọi module, ghi cấu hình, biểu đồ và nhận xét. Dùng **bảy notebook** trong `notebooks/`:

1. `01_data_exploration.ipynb` — khảo sát nguồn dữ liệu, thống kê, chốt lọc và split; local/Kaggle CPU.
2. `02_tokenizer_data_preparation.ipynb` — huấn luyện và kiểm tra BPE, tạo dữ liệu pretrain/SFT đã token hóa; Kaggle CPU.
3. `03_model_sanity_checks.ipynb` — kiến trúc, số tham số, forward/backward, overfit một batch, cache và checkpoint/resume; local GPU/T4.
4. `04_pretraining.ipynb` — pilot trong cùng run, pretrain, loss và checkpoint; Kaggle T4.
5. `05_kv_cache_profiling.ipynb` — nạp checkpoint pretrain, đo cache ON/OFF và xuất báo cáo; Kaggle T4.
6. `06_math_sft.ipynb` — nạp checkpoint pretrain, SFT và chọn checkpoint bằng validation; Kaggle T4.
7. `07_reasoning_evaluation.ipynb` — nạp checkpoint SFT, greedy, voting, Best-of-N, đánh giá và phân tích lỗi; Kaggle T4.

Thứ tự chạy là 01 → 02 → 03 → 04 → 05 → 06 → 07. Mỗi notebook chạy độc lập trong phiên mới bằng artifact đã lưu, không phụ thuộc biến bộ nhớ của notebook trước. Notebook 06 tiếp tục từ trọng số notebook 04 đã xuất, không chạy lại pretrain; profiling không sửa checkpoint. Manifest, tokenizer, checkpoint và log được lưu bền vững để chuyển phiên. Chi tiết đầu vào/đầu ra và task tương ứng nằm trong `KE_HOACH_THUC_HIEN.md`.

Module gợi ý: `tokenizer/`, `model/`, `data/`, `training/`, `inference/`, `evaluation/`, `tests/`, `configs/`.

Dùng `pyproject.toml` và `uv.lock` cho môi trường local. Trên Kaggle, kiểm tra Python, PyTorch và CUDA trước khi cài dependencies; không thay bản PyTorch có CUDA đang chạy chỉ để đồng bộ lockfile. Ghi các phiên bản thực tế, seed, commit Git, hash hoặc phiên bản dữ liệu, tham số huấn luyện và đường dẫn checkpoint. `README.md` có lệnh chạy từ chuẩn bị dữ liệu đến đánh giá; không commit dữ liệu lớn, checkpoint hoặc thông tin xác thực vào Git.

Chọn **TensorBoard** để lưu train/validation loss của cả pretrain và SFT; bàn giao event files, biểu đồ và log dạng bảng. README có **một lệnh tái lập đánh giá từ checkpoint cuối**, ghi rõ artifact, môi trường và cách lấy dữ liệu/checkpoint; quy trình train từ đầu và resume được hướng dẫn riêng. Demo terminal có một lệnh nạp checkpoint, chạy forward và sinh tự hồi quy; demo tương tác cuối cho phép nhập câu hỏi, chọn greedy/voting/Best-of-N và xem token/thời gian.

## 8. Lịch triển khai 12 ngày

| Ngày | Kết quả bắt buộc |
|---|---|
| 1–2 | Repository, cấu hình, pipeline dữ liệu mẫu, BPE và kiểm tra tokenizer. |
| 3–4 | Transformer, forward/backward, causal mask, RoPE, greedy generation, KV cache, kiểm tra cached/non-cached, lưu/resume và chuẩn bị script profiling. |
| 5–6 | Pilot trong cùng run, pretrain theo mốc thời gian, checkpoint và log; demo nạp checkpoint pretrain và profiling chính thức cache ON/OFF trên T4. |
| 7 | SFT Orca Math + GSM8K train, chọn checkpoint theo validation; bàn giao TensorBoard logs và biểu đồ loss cả hai giai đoạn. |
| 8–9 | Rà báo cáo KV cache từ checkpoint pretrain; bộ trích đáp số, self-consistency và verifier trên validation bằng checkpoint SFT. |
| 10 | Khóa cấu hình, chạy đánh giá test; lưu dự đoán, chi phí và ít nhất năm ca sai. |
| 11–12 | Báo cáo IEEE conference hai cột 6–8 trang, README với lệnh tái lập đánh giá, demo terminal tương tác, slide và rà kết quả tái lập. |

Viết mô tả phương pháp, sơ đồ kiến trúc và bảng cấu hình trong khi phát triển; không dồn toàn bộ báo cáo sang ngày cuối.

## 9. Tiêu chí nghiệm thu

1. Tokenizer tự xây encode/decode đúng trên văn bản và ký hiệu toán; có file lưu và nạp lại.
2. Transformer tự viết chạy forward/backward, loss giảm trên một batch nhỏ, có số tham số được đo.
3. Causal mask ngăn nhìn token tương lai; cache giữ logits/greedy gần tương đương với chế độ không cache.
4. Có checkpoint pretrain và checkpoint SFT từ **cùng một chuỗi huấn luyện**, TensorBoard event files, biểu đồ loss và số token thực tế; demo nạp trọng số và sinh tự hồi quy chạy được.
5. Có báo cáo profiling chính thức dùng checkpoint pretrain trên T4: bảng/biểu đồ cache ON/OFF tại L = 128, 512, 1.024, 2.048 với latency, throughput, VRAM và cấu hình đo. Trước train phải kiểm tra cache đúng; không bắt buộc đo lại sau SFT.
6. Có script đánh giá GSM8K, báo cáo greedy, Pass@k, majority voting và Best-of-N; không dùng test để tinh chỉnh.
7. Có phân tích ít nhất năm ca thất bại, giới hạn của verifier và giới hạn do mô hình nhỏ/ít token.
8. README có một lệnh tái lập đánh giá từ checkpoint cuối và hướng dẫn train/resume; báo cáo IEEE conference hai cột 6–8 trang, demo terminal tương tác và slide.

## 10. Rủi ro và phương án xử lý

| Rủi ro | Cách xử lý |
|---|---|
| Throughput không đủ 100M token | Giảm ngân sách token theo tốc độ pilot; bảo vệ thời gian SFT và đánh giá. |
| Hết VRAM | Giảm micro-batch, dùng accumulation; đo lại throughput thay vì giả định nhanh hơn. |
| Mất phiên Kaggle | Checkpoint định kỳ đầy đủ, kiểm tra resume, xuất checkpoint ra nơi lưu trữ bền vững. |
| GSM8K accuracy rất thấp | Báo cáo trung thực; phân tích Pass@k, tỷ lệ đáp số hợp lệ và lỗi lời giải; không diễn giải cải thiện từ mẫu nhỏ quá mức. |
| Verifier không phân biệt được đúng/sai | Công bố coverage và lỗi; dùng kết quả âm tính như một phát hiện, không dùng nhãn test để sửa verifier. |
| Dữ liệu trùng benchmark | Hash và so khớp câu hỏi đã chuẩn hóa với GSM8K test trong phạm vi khả thi; ghi rõ giới hạn của kiểm tra. |

## 11. Sản phẩm bàn giao

Repository chứa mã nguồn PyTorch, tokenizer, cấu hình, bảy notebook thí nghiệm, kiểm tra cần thiết, README, script đánh giá và profiling. Kết quả gồm checkpoint pretrain/SFT, TensorBoard event files và log dạng bảng, báo cáo profiling chính thức từ checkpoint pretrain, bảng accuracy–compute, dự đoán và phân tích lỗi. Báo cáo khoa học **IEEE conference hai cột, 6–8 trang**, theo cấu trúc Abstract, Method, Experimental Setup, Results, Discussion, Ethics & Limitations; kèm slide và demo terminal tương tác. README cung cấp một lệnh tái lập đánh giá từ checkpoint cuối cùng các bước chuẩn bị môi trường và artifact.

## 12. Kết luận

Đồ án xây dựng **một mô hình khoảng 36M tham số từ trọng số ngẫu nhiên**, pretrain trên tối đa khoảng 100M token văn bản giáo dục và toán, sau đó SFT trên dữ liệu toán có lời giải. Trọng tâm đóng góp là hệ thống tự triển khai có thể kiểm chứng và thí nghiệm đo quan hệ giữa độ chính xác suy luận, số ứng viên và chi phí inference. Mọi kết luận về năng lực mô hình phải dựa trên kết quả thực đo; bối cảnh 12 ngày và một T4 được ghi rõ như giới hạn nghiên cứu.
