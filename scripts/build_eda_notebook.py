"""Update notebook 01 with reproducible dataset exploration cells."""

import json
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "notebooks/01_data_exploration.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))
setup = "".join(notebook["cells"][1]["source"])
setup = setup.replace("from datasets import load_dataset", "from datasets import load_dataset, load_dataset_builder")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


notebook["cells"] = [
    md("""# 01 — Khai phá dữ liệu (EDA)

Mục tiêu: hiểu **quy mô, schema, chất lượng và độ dài** trước khi chuẩn bị dữ liệu.
FineWeb-Edu và OpenWebMath dùng cho pretrain; **chỉ GSM8K train dùng cho SFT**.
Không khảo sát nội dung GSM8K test để chọn bộ lọc hoặc cấu hình.

Notebook chạy trên CPU. Quy mô nguồn lấy từ metadata tại revision cố định; nếu metadata thiếu thì ghi `None`, không suy ra tổng quy mô từ mẫu.
Thống kê chất lượng pretrain chỉ áp dụng cho mẫu có shuffle giới hạn bộ đệm, chưa phải mẫu ngẫu nhiên đều của toàn corpus.
"""),
    code(setup),
    md("""## 1. Quy mô nguồn và phạm vi khảo sát

Phân biệt ba đại lượng: **quy mô nguồn**, **ngân sách dự án** (80M/20M token), và **số mẫu thực sự khảo sát**.
FineWeb-Edu dùng subset `sample-10BT`, không phải toàn bộ FineWeb-Edu.
Số token nguồn dùng tokenizer khác nên không đồng nhất với số token BPE của dự án.
"""),
    code('''import html, math, statistics
from collections import Counter
from IPython.display import display, HTML
from llm_reasoning.data_prep import content_key
from llm_reasoning.sft_data import extract_gsm_answer

SAMPLE_SIZE = 1000
SHUFFLE_BUFFER = 5000
SEED = 42
assert SAMPLE_SIZE > 0 and SHUFFLE_BUFFER > 0
ROOT = Path('/kaggle/working/llm-reasoning') if ON_KAGGLE else REPO_DIR / 'artifacts'
EDA_DIR = ROOT / 'eda'
EDA_DIR.mkdir(parents=True, exist_ok=True)

def table(rows):
    if not rows:
        return
    keys = list(rows[0])
    heading = ''.join('<th style="padding:6px">' + html.escape(str(k)) + '</th>' for k in keys)
    body = ''.join('<tr>' + ''.join('<td style="padding:6px">' + html.escape(str(row.get(k))) + '</td>' for k in keys) + '</tr>' for row in rows)
    display(HTML('<table border="1"><thead><tr>' + heading + '</tr></thead><tbody>' + body + '</tbody></table>'))

sources = {
    'fineweb': (config.data.fineweb_repo, 'sample-10BT', 'text'),
    'math': (config.data.math_repo, None, 'text'),
    'gsm8k': (config.data.gsm8k_repo, 'main', 'question'),
}
api = HfApi()
revisions, inventory, samples = {}, [], {}
for label, (repo, subset, field) in sources.items():
    revision = api.repo_info(repo_id=repo, repo_type='dataset').sha
    revisions[label] = revision
    # Builder reads metadata; it does not download/prepare the full corpus.
    builder = load_dataset_builder(repo, name=subset, revision=revision)
    info = builder.info
    inventory.append({
        'source': label, 'repository': repo, 'subset': builder.config.name,
        'revision': revision,
        'split_rows_metadata': {k: v.num_examples for k, v in (info.splits or {}).items()},
        'download_GiB_metadata': round(info.download_size / 2**30, 3) if info.download_size is not None else None,
        'schema': str(info.features),
        'project_budget': '80M BPE tokens' if label == 'fineweb' else '20M BPE tokens' if label == 'math' else 'train minus 500 validation questions',
    })
    if label == 'gsm8k':
        samples[label] = list(load_dataset(repo, name=subset, split='train', revision=revision))
    else:
        stream = load_dataset(repo, name=subset, split='train', revision=revision, streaming=True)
        samples[label] = list(stream.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER).take(SAMPLE_SIZE))
table(inventory)
table([{'source': k, 'examined_rows': len(v), 'scope': 'all train' if k == 'gsm8k' else 'buffer-shuffled sample'} for k, v in samples.items()])
'''),
    md("""## 2. Dữ liệu thiếu, trùng và độ dài văn bản

Đếm ô thiếu/rỗng, trùng theo hash văn bản chuẩn hóa, ký tự thay thế Unicode và marker HTML còn sót.
HTML marker chỉ là tín hiệu cần xem mẫu, không phải kết luận một tài liệu lỗi.
Với GSM8K, kiểm tra cả câu hỏi và lời giải, tỷ lệ đọc được đáp số `####`.
"""),
    code('''import re

def length_stats(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return {'count': len(values), 'min': values[0], 'mean': round(statistics.mean(values), 2),
            'p50': values[(len(values)-1)//2], 'p95': values[int((len(values)-1)*.95)],
            'p99': values[int((len(values)-1)*.99)], 'max': values[-1]}

quality, lengths, text_lengths = [], [], {}
for label, (_, _, field) in sources.items():
    rows = samples[label]
    fields = [field, 'answer'] if label == 'gsm8k' else [field]
    for name in fields:
        texts = [r.get(name) if isinstance(r.get(name), str) else '' for r in rows]
        nonempty = [t for t in texts if t.strip()]
        counts = Counter(content_key(t) for t in nonempty)
        quality.append({'source': label, 'field': name, 'examined': len(rows),
                        'missing_or_empty': len(rows)-len(nonempty),
                        'duplicate_rows_after_first': sum(n-1 for n in counts.values()),
                        'replacement_char_rows': sum(chr(0xfffd) in t for t in texts),
                        'html_marker_rows': sum(bool(re.search(r'<(?:html|script|div|body)\\b', t, re.I)) for t in texts)})
        values = [len(t) for t in texts]
        text_lengths[label + '.' + name] = values
        lengths.append({'source': label, 'field': name, 'unit': 'characters', **length_stats(values)})
table(quality)
table(lengths)
gsm_parsed = [extract_gsm_answer(r['answer']) for r in samples['gsm8k']]
print('GSM8K answer extraction:', sum(x is not None for x in gsm_parsed), '/', len(gsm_parsed))

# Show examples instead of inferring semantic quality from length alone.
for label, (_, _, field) in sources.items():
    display(HTML('<h4>' + html.escape(label) + ': short / median / long examples</h4>'))
    ordered = sorted(samples[label], key=lambda r: len(str(r.get(field) or '')))
    if ordered:
        table([{'position': tag, 'characters': len(str(row.get(field) or '')),
                'text_preview': str(row.get(field) or '')[:700]}
               for tag, row in [('short', ordered[0]), ('median', ordered[len(ordered)//2]), ('long', ordered[-1])]])
'''),
    md("""## 3. Biểu đồ phân bố độ dài

Biểu đồ dùng các khoảng tăng theo log để thấy cả chuỗi ngắn và chuỗi rất dài. Trục dọc là số bản ghi trong phạm vi khảo sát.
"""),
    code('''def histogram_svg(values, title):
    edges = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, math.inf]
    counts = [sum(a <= x < b for x in values) for a, b in zip(edges, edges[1:])]
    peak = max(counts, default=1) or 1
    bars = []
    for i, count in enumerate(counts):
        x, height = 40 + i*70, 180*count/peak
        label = str(edges[i]) + '+' if math.isinf(edges[i+1]) else str(edges[i]) + '-' + str(edges[i+1]-1)
        bars.append(f'<rect x="{x}" y="{230-height}" width="55" height="{height}" fill="#377eb8"/><text x="{x+27}" y="{220-height}" text-anchor="middle">{count}</text><text x="{x+27}" y="250" text-anchor="middle" font-size="9">{label}</text>')
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 285"><rect width="700" height="285" fill="white"/><text x="20" y="24">' + html.escape(title) + '</text>' + ''.join(bars) + '<text x="250" y="275">Length (characters); bar labels = record counts</text></svg>'
    return svg

for label, values in text_lengths.items():
    svg = histogram_svg(values, label)
    (EDA_DIR / (label + '.svg')).write_text(svg, encoding='utf-8')
    display(HTML(svg))
'''),
    md("""## 4. GSM8K: chia tập và tác động lên SFT

Phân tích toàn bộ train, giữ 500 câu validation theo hash/seed như pipeline. Không đọc nội dung test ở bước này.
Nếu đã có tokenizer dự án, đo số token và tỷ lệ vượt 512/1024; nếu chưa có thì bỏ qua, không quy đổi ký tự thành token.
Đây là phép đo theo tokenizer hiện có: phải đo lại khi thay tokenizer chính thức.
"""),
    code('''from llm_reasoning.data_prep import split_gsm8k
from llm_reasoning.sft_data import encode_example
from llm_reasoning.tokenizer import ByteBPETokenizer

split = split_gsm8k(samples['gsm8k'], [], revision=revisions['gsm8k'], seed=SEED, validation_count=500)
table([{'split': 'gsm_train_before_filter', 'rows': len(split.train)},
       {'split': 'gsm_validation', 'rows': len(split.validation)}])
token_summary = []
tokenizer_path = ROOT / 'data' / 'pretrain' / 'tokenizer.json'
if tokenizer_path.exists():
    import hashlib
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    print('Tokenizer SHA256:', hashlib.sha256(tokenizer_path.read_bytes()).hexdigest())
    for name, rows in [('train', split.train), ('validation', split.validation)]:
        sizes, short, invalid = [], 0, 0
        for row in rows:
            parsed = extract_gsm_answer(row['answer'])
            if parsed is None:
                invalid += 1
                continue
            solution, answer = parsed
            if len(solution.split()) < 4:
                short += 1
                continue
            example = encode_example(tokenizer, source='gsm8k', question=row['question'],
                                     solution=solution, answer=answer, max_seq_len=10**12)
            sizes.append(example['token_count'])
        token_summary.append({'split': name, 'unit': 'project BPE tokens', **length_stats(sizes),
                              'invalid_answer': invalid, 'too_short_solution': short,
                              'over_512': sum(x > 512 for x in sizes), 'over_1024': sum(x > 1024 for x in sizes),
                              'accepted_tokens_at_1024': sum(x for x in sizes if x <= 1024)})
    table(token_summary)
else:
    print('No project tokenizer yet: run notebook 02, then rerun this section for actual BPE lengths.')
'''),
    md("""## 5. Lưu bằng chứng và kết luận review

File JSON chứa revision, phạm vi khảo sát và các thống kê. SVG có thể dùng trong báo cáo.
Khi đọc kết quả, ghi nhận:

- Quy mô subset nguồn và số tài liệu khảo sát có khác nhau không?
- Dữ liệu có bản rỗng, trùng hoặc trang toán lạc đề không? Xem mẫu trước khi đổi bộ lọc.
- p95/p99 có cho thấy nhiều chuỗi dài? Bao nhiêu câu GSM8K bị loại tại giới hạn 1024 token?
- Ngân sách 80M/20M token là mục tiêu chuẩn bị; số thực tế phải lấy từ manifest notebook 02.
- EDA không chứng minh lời giải đúng về toán học; không dùng test để sửa dữ liệu hoặc chọn cấu hình.
"""),
    code('''summary = {'seed': SEED, 'sample_size_limit_pretrain': SAMPLE_SIZE,
           'shuffle_buffer': SHUFFLE_BUFFER, 'inventory': inventory,
           'examined_rows': {k: len(v) for k, v in samples.items()},
           'quality': quality, 'character_lengths': lengths,
           'gsm_answer_parse_success': sum(x is not None for x in gsm_parsed),
           'gsm_token_lengths': token_summary}
summary_path = EDA_DIR / 'summary.json'
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')
print('Saved:', summary_path)
print('Charts:', EDA_DIR)
'''),
]
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
