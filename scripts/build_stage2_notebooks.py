"""Build reviewable stage-2 notebooks from one checked-in source file."""

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'notebooks'


def markdown(source):
    return dict(cell_type='markdown', metadata={}, source=textwrap.dedent(source).strip().splitlines(True))


def code(source):
    return dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                source=textwrap.dedent(source).strip().splitlines(True))


def write(name, cells):
    notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3',
        'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=5)
    (DEST / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


SETUP = r'''
from pathlib import Path
import json, os, subprocess, sys
import torch

REPO_URL = 'https://github.com/ngochhuy/llm-reasoning.git'
REPO_REF = ''  # Kaggle: đặt commit SHA đã review; không dùng main di động.
ON_KAGGLE = Path('/kaggle/working').exists()
REPO_DIR = Path('/kaggle/working/llm-reasoning-repo') if ON_KAGGLE else next(
    (p for p in (Path.cwd(), Path.cwd().parent) if (p/'src'/'llm_reasoning').exists()), Path.cwd())
if ON_KAGGLE:
    if len(REPO_REF) < 40:
        raise ValueError('Đặt REPO_REF bằng commit SHA 40 ký tự đã review trước khi chạy Kaggle')
    if not (REPO_DIR/'src'/'llm_reasoning').exists():
        subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', '--detach', REPO_REF], check=True)
CODE_COMMIT = subprocess.check_output(['git', '-C', str(REPO_DIR), 'rev-parse', 'HEAD'], text=True).strip()
DIRTY = bool(subprocess.check_output(['git', '-C', str(REPO_DIR), 'status', '--porcelain'], text=True).strip())
print('commit:', CODE_COMMIT, 'dirty:', DIRTY, 'torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
sys.path.insert(0, str(REPO_DIR/'src'))
os.environ['PYTHONPATH'] = str(REPO_DIR/'src') + os.pathsep + os.environ.get('PYTHONPATH', '')
'''


write('03_model_sanity_checks.ipynb', [
    markdown('''# 03 — Kiểm tra mô hình trước huấn luyện

    Chạy độc lập trên máy local hoặc Kaggle. Các bài thử dùng trọng số ngẫu nhiên và dữ liệu tổng hợp, **không huấn luyện mô hình chính**. Trên Kaggle đặt `REPO_REF` bằng SHA đã review. Cần Python 3.11, PyTorch, pytest, TensorBoard; local: `uv sync --group model --group train --group dev --group notebook`. Kaggle: kiểm tra PyTorch/CUDA cài sẵn rồi cài thiếu `pytest tensorboard ipykernel`, không thay torch CUDA bằng wheel CPU.'''),
    code(SETUP),
    code(r'''
import importlib.util
for package in ('pytest', 'tensorboard'):
    if importlib.util.find_spec(package) is None:
        raise RuntimeError(f'Cần cài {package} rồi chạy lại kernel')
if ON_KAGGLE and DIRTY:
    raise RuntimeError('Repo Kaggle phải sạch và đúng commit SHA')
from llm_reasoning.config import make_config
from llm_reasoning.model import TransformerLM
from llm_reasoning.tokenizer import ByteBPETokenizer
cfg = make_config('target')
print('Target config:', cfg.model, cfg.train)
print('Target parameters at vocab 8000:', TransformerLM(cfg.model, 8000).num_parameters())
'''),
    markdown('''## Kiểm thử cơ chế

    Forward/backward, causal mask, tokenizer, cache, training và checkpoint/resume. Kết quả phải xanh trước notebook 04. CPU FP16/CUDA tests được skip trên máy không có CUDA.'''),
    code(r'''
tests = ['tests/test_layers.py', 'tests/test_attention.py', 'tests/test_model.py',
         'tests/test_generation.py', 'tests/test_kv_cache.py', 'tests/test_training.py',
         'tests/test_checkpoint.py', 'tests/test_training_data.py']
completed = subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests], cwd=REPO_DIR, check=True)
print('Test return code:', completed.returncode)
'''),
    markdown('''## Overfit và resume trên dữ liệu tổng hợp

    Mỗi script xuất summary, metrics CSV và TensorBoard trong thư mục mới. Script resume so sánh run liền và run ngắt trên cùng CPU. Đây là hai bài kiểm tra độc lập, không phải hai mô hình ứng viên.'''),
    code(r'''
from datetime import datetime, timezone
OUT = (Path('/kaggle/working') if ON_KAGGLE else REPO_DIR/'artifacts') / ('nb03_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
subprocess.run([sys.executable, str(REPO_DIR/'scripts/task_2_6_smoke.py'),
                '--output', str(OUT/'overfit')], cwd=REPO_DIR, check=True)
subprocess.run([sys.executable, str(REPO_DIR/'scripts/task_2_7_smoke.py'),
                '--output', str(OUT/'resume')], cwd=REPO_DIR, check=True)
summary = json.loads((OUT/'resume'/'summary.json').read_text(encoding='utf-8'))
print('Output:', OUT, '\nResume:', json.dumps(summary, indent=2))
'''),
])


write('04_pretraining.ipynb', [
    markdown('''# 04 — Pretrain một mô hình từ đầu

    Chạy độc lập trên Kaggle T4 sau review giai đoạn 2. `RUN_MODE='inspect'` là mặc định, không train. `smoke` chạy tối đa 2 update bằng model nhỏ trên shard thật; `production` mới khởi tạo khoảng 36M tham số và chạy tối đa 1 epoch pretrain. Luôn dùng tokenizer và shard cùng một lần chuẩn bị dữ liệu. Local: `uv sync --group model --group train --group data --group notebook`. Kaggle: xác minh torch CUDA cài sẵn; cài thiếu `tensorboard numpy` rồi khởi động lại kernel. Không cài lại torch bằng wheel CPU.'''),
    code(SETUP),
    code(r'''
from dataclasses import asdict, replace
from llm_reasoning.checkpoint import file_sha256, load_checkpoint, save_checkpoint
from llm_reasoning.config import make_config
from llm_reasoning.data_stream import ResumableBatchStream
from llm_reasoning.model import TransformerLM
from llm_reasoning.tokenizer import ByteBPETokenizer
from llm_reasoning.training import Trainer, TrainingLogger
from llm_reasoning.training_data import PretrainBatches
from llm_reasoning.preflight import inspect as run_preflight
import time

RUN_MODE = 'inspect'  # inspect | smoke | production; bật production sau review.
QUOTA_CONFIRMED = False  # chỉ True sau khi tự kiểm tra GPU hours/session limit trong Kaggle.
if ON_KAGGLE:
    candidates = [path.parent for path in Path('/kaggle/input').rglob('manifest.json')
                  if path.parent.name == 'pretrain' and (path.parent/'tokenizer.json').is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f'Cần gắn đúng một dataset pretrain vào Kaggle Input; tìm thấy: {candidates}')
    DATA_ROOT = candidates[0]
else:
    DATA_ROOT = REPO_DIR/'artifacts/data/pretrain'
print('Pretrain data:', DATA_ROOT)
OUTPUT_ROOT = Path('/kaggle/working/pretrain_run') if ON_KAGGLE else REPO_DIR/'artifacts/nb04_pretrain'
RESUME_FROM = None  # Path đến latest.pt đã lưu bền vững; giữ đúng dataset, commit và thiết bị.
MAX_UPDATES_THIS_SESSION = None  # Pilot: 200-500, rồi resume cùng checkpoint và total_steps.
if RUN_MODE not in ('inspect', 'smoke', 'production'):
    raise ValueError('RUN_MODE không hợp lệ')
if RUN_MODE == 'production' and (not ON_KAGGLE or not torch.cuda.is_available() or DIRTY):
    raise RuntimeError('Production yêu cầu Kaggle CUDA và commit sạch đã chốt')
if RESUME_FROM is not None and RUN_MODE != 'production':
    raise ValueError('Resume chỉ dùng cho cùng production run')
tokenizer_path = DATA_ROOT/'tokenizer.json'
manifest_path = DATA_ROOT/'manifest.json'
tokenizer = ByteBPETokenizer.load(tokenizer_path)
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
if manifest['tokenizer_sha256'] != file_sha256(tokenizer_path) or manifest['vocab_size'] != tokenizer.vocab_size:
    raise ValueError('Tokenizer hash/vocab không khớp manifest')
if manifest['block_size'] != 512 and RUN_MODE == 'production':
    raise ValueError('Production yêu cầu pretrain block size 512')
if RUN_MODE == 'production':
    for source, target in (('fineweb', 80_000_000), ('math', 20_000_000)):
        if manifest['sources'][source]['shards']['train']['written_tokens'] < target - 512:
            raise ValueError(f'{source} chưa đạt ngân sách token pretrain; tạo lại dữ liệu trước GPU run')
print('Mode:', RUN_MODE, 'tokenizer SHA:', file_sha256(tokenizer_path), 'vocab:', tokenizer.vocab_size)
print('Source train tokens:', {s: d['shards']['train']['written_tokens'] for s,d in manifest['sources'].items()})
preflight = run_preflight(DATA_ROOT.parent, OUTPUT_ROOT, REPO_DIR,
                          production=ON_KAGGLE, expected_commit=REPO_REF if ON_KAGGLE else None,
                          quota_confirmed=QUOTA_CONFIRMED, require_sft=False)
print('Preflight:', [(item['name'], item['status']) for item in preflight['checks']])
if RUN_MODE == 'production' and not preflight['ready']:
    raise RuntimeError('Preflight chưa đạt; xem các mục fail/pending trước GPU run')
'''),
    markdown('''## Kiểm tra batch và tính lịch train

    Tất cả batch đọc từ shard uint16 do notebook 02 tạo. Fingerprint bao gồm manifest, shard và cách chia batch; resume từ dữ liệu khác sẽ bị từ chối. `smoke` chỉ dùng vài batch và checkpoint thử, không được dùng làm pretrain chính.'''),
    code(r'''
if RUN_MODE != 'inspect':
    config = make_config('smoke' if RUN_MODE == 'smoke' else 'target')
    train_cfg = config.train
    if RUN_MODE == 'smoke':
        train_cfg = replace(train_cfg, pretrain_seq_len=manifest['block_size'],
                            micro_batch_size=1, gradient_accumulation_steps=1)
        model_cfg = replace(config.model, max_context_tokens=max(1024, manifest['block_size']))
    else:
        model_cfg = config.model
    if train_cfg.pretrain_seq_len != manifest['block_size']:
        raise ValueError('Block size khác pretrain_seq_len')
    train_data = PretrainBatches(DATA_ROOT, 'train', batch_size=train_cfg.micro_batch_size,
                                 max_batches=4 if RUN_MODE == 'smoke' else None)
    stream = ResumableBatchStream(train_data, fingerprint=train_data.fingerprint,
                                  seed=config.data.seed, shuffle=True)
    total_steps = (len(train_data) + train_cfg.gradient_accumulation_steps - 1) // train_cfg.gradient_accumulation_steps
    device = 'cuda' if RUN_MODE == 'production' else 'cpu'
    if RUN_MODE == 'smoke':
        train_cfg = replace(train_cfg, precision='fp32')
    print('Microbatches:', len(train_data), 'total steps:', total_steps,
          'batch:', train_data[0]['input_ids'].shape, 'fingerprint:', train_data.fingerprint)
else:
    print('Inspect hoàn tất; chưa khởi tạo model/trainer, chưa train.')
'''),
    markdown('''## Train, checkpoint và validation

    Chạy lại từ phiên mới: tải `latest.pt`, cùng tokenizer/shard vào Kaggle Input; đặt `RESUME_FROM` và giữ nguyên cấu hình batch/dataset. `MAX_UPDATES_THIS_SESSION` dừng ở biên update; có thể dùng 200–500 update để đo pilot, rồi resume từ `latest.pt`. Checkpoint cuối luôn lưu sau mỗi phiên. Copy toàn bộ output sang Kaggle Dataset/Notebook Version trước khi phiên kết thúc.'''),
    code(r'''
if RUN_MODE != 'inspect':
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_ROOT/'latest.pt'
    log_dir = OUTPUT_ROOT/'logs'
    if RESUME_FROM is None:
        if checkpoint_path.exists() or (log_dir/'metrics.csv').exists():
            raise FileExistsError('Output cũ tồn tại; chọn thư mục mới hoặc đặt RESUME_FROM')
        torch.manual_seed(config.data.seed)
        model = TransformerLM(model_cfg, tokenizer.vocab_size).to(device)
        trainer = Trainer(model, train_cfg, stage='pretrain', total_steps=total_steps)
    else:
        trainer = load_checkpoint(RESUME_FROM, stream, tokenizer_path=tokenizer_path,
                                  device=device, expected_stage='pretrain')
        if trainer.total_steps != total_steps or trainer.config != train_cfg:
            raise ValueError('Config hoặc total_steps thay đổi sau resume')
    kind = 'smoke' if RUN_MODE == 'smoke' else 'training'
    run_manifest = dict(stage='pretrain', kind=kind, code_commit=CODE_COMMIT,
                        tokenizer_sha256=file_sha256(tokenizer_path),
                        data_fingerprint=train_data.fingerprint, model_config=asdict(model_cfg),
                        train_config=asdict(train_cfg), total_steps=total_steps, device=device)
    run_manifest_path = OUTPUT_ROOT/'run_manifest.json'
    if run_manifest_path.exists():
        if json.loads(run_manifest_path.read_text(encoding='utf-8')) != run_manifest:
            raise ValueError('Run manifest thay đổi; không thể resume cùng output')
    else:
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2)+'\n', encoding='utf-8')
    with TrainingLogger(log_dir, resume_step=trainer.state.attempted_steps if RESUME_FROM else None) as logger:
        trainer.logger = logger
        last_saved = time.monotonic()
        def periodic(current):
            global last_saved
            if time.monotonic() - last_saved >= train_cfg.checkpoint_interval_minutes * 60:
                save_checkpoint(checkpoint_path, current, stream,
                                tokenizer_path=tokenizer_path, checkpoint_kind=kind)
                last_saved = time.monotonic()
        limit = 2 if RUN_MODE == 'smoke' else MAX_UPDATES_THIS_SESSION
        rows = trainer.train_epoch(stream, max_updates=limit, on_update=periodic)
        save_checkpoint(checkpoint_path, trainer, stream,
                        tokenizer_path=tokenizer_path, checkpoint_kind=kind)
        print('New updates:', len(rows), 'total:', trainer.state.attempted_steps,
              'processed tokens:', trainer.state.processed_tokens, 'checkpoint:', checkpoint_path)
        if stream.cursor == len(stream.batches):
            validation = PretrainBatches(DATA_ROOT, 'validation', batch_size=train_cfg.micro_batch_size,
                                         max_batches=2 if RUN_MODE == 'smoke' else None)
            print('Validation:', trainer.evaluate(validation))
    print('Train xong phiên này. Lưu checkpoint và logs trước khi đóng phiên Kaggle.')
'''),
])


write('05_kv_cache_profiling.ipynb', [
    markdown('''# 05 — Profiling KV cache ON/OFF

    Chạy phiên mới sau notebook 04. Mặc định `RUN_MODE='inspect'`, chỉ xác minh checkpoint. `smoke` đo nhanh checkpoint thử. `official` dùng checkpoint **pretrain chính** trên T4: prompt 128/512/1024/2048, sinh 128 token, 5 lần đo và warmup 2. Output JSON, CSV, PNG, PDF, report.md. Local: `uv sync --group model --group profile --group notebook`. Kaggle cần `matplotlib`; giữ torch CUDA hiện có.'''),
    code(SETUP),
    code(r'''
from llm_reasoning.checkpoint import file_sha256, load_model_artifact
from llm_reasoning.tokenizer import ByteBPETokenizer
RUN_MODE = 'inspect'  # inspect | smoke | official
APPROVED_EARLY_STOP = False  # chỉ bật sau review checkpoint dừng sớm vì giới hạn thời gian.
CHECKPOINT = Path('/kaggle/input/llm-reasoning-pretrain/latest.pt') if ON_KAGGLE else REPO_DIR/'artifacts/nb04_pretrain/latest.pt'
TOKENIZER = Path('/kaggle/input/llm-reasoning-data/pretrain/tokenizer.json') if ON_KAGGLE else REPO_DIR/'artifacts/data/pretrain/tokenizer.json'
OUTPUT = Path('/kaggle/working/kv_profile') if ON_KAGGLE else REPO_DIR/'artifacts/nb05_profile'
if RUN_MODE not in ('inspect', 'smoke', 'official'):
    raise ValueError('RUN_MODE không hợp lệ')
if RUN_MODE == 'official' and (not ON_KAGGLE or not torch.cuda.is_available() or DIRTY):
    raise RuntimeError('Profiling chính thức yêu cầu T4/CUDA và commit sạch')
_, meta = load_model_artifact(CHECKPOINT, tokenizer_path=TOKENIZER)
if meta['stage'] != 'pretrain':
    raise ValueError('Notebook 05 yêu cầu checkpoint pretrain')
if RUN_MODE == 'official' and meta['checkpoint_kind'] != 'training':
    raise ValueError('Checkpoint này chỉ là smoke/unclassified, không dùng đo chính thức')
if RUN_MODE == 'official' and not APPROVED_EARLY_STOP and meta['train_state']['attempted_steps'] < meta['total_steps']:
    raise ValueError('Checkpoint pretrain còn dở; hoàn thành hoặc ghi rõ đây là đo pilot')
print('Checkpoint:', CHECKPOINT, 'SHA:', file_sha256(CHECKPOINT))
print('Tokenizer SHA:', file_sha256(TOKENIZER), 'metadata:', meta)
'''),
    code(r'''
if RUN_MODE != 'inspect':
    if OUTPUT.exists():
        raise FileExistsError('Output profiling tồn tại; chọn thư mục mới')
    command = [sys.executable, '-m', 'llm_reasoning.profiling', '--checkpoint', str(CHECKPOINT),
               '--tokenizer', str(TOKENIZER), '--output', str(OUTPUT),
               '--device', 'cuda' if RUN_MODE == 'official' else 'cpu',
               '--dtype', 'float16' if RUN_MODE == 'official' else 'checkpoint',
               '--prompt-lengths', *map(str, (128, 512, 1024, 2048) if RUN_MODE == 'official' else (16, 32)),
               '--new-tokens', '128' if RUN_MODE == 'official' else '8',
               '--repetitions', '5' if RUN_MODE == 'official' else '2',
               '--warmup', '2' if RUN_MODE == 'official' else '1']
    subprocess.run(command, cwd=REPO_DIR, check=True)
    result = json.loads((OUTPUT/'results.json').read_text(encoding='utf-8'))
    print('Greedy pairs match:', result['all_pairs_match'], 'report:', OUTPUT/'report.md')
    if not result['all_pairs_match']:
        raise RuntimeError('KV cache ON/OFF không khớp')
else:
    print('Inspect hoàn tất; chưa chạy đo.')
'''),
])


write('06_math_sft.ipynb', [
    markdown('''# 06 — GSM8K SFT từ checkpoint pretrain

    Chạy phiên mới, không chạy lại notebook 04. `RUN_MODE='inspect'` mặc định. `smoke` dùng mô hình/checkpoint pretrain thử, bốn microbatch để kiểm cả validation và snapshot; `production` nạp **trọng số pretrain chính**, tạo optimizer/lịch SFT mới, train 1 epoch trước khi xem validation và quyết định có tiếp tục tối đa 3 epoch. Không dùng Orca. Local: `uv sync --group model --group train --group data --group notebook`; Kaggle cần `tensorboard numpy`, giữ torch CUDA cài sẵn.'''),
    code(SETUP),
    code(r'''
from dataclasses import asdict, replace
from llm_reasoning.checkpoint import file_sha256, load_checkpoint, load_model_artifact, save_checkpoint
from llm_reasoning.config import make_config
from llm_reasoning.data_stream import ResumableBatchStream
from llm_reasoning.tokenizer import ByteBPETokenizer
from llm_reasoning.training import Trainer, TrainingLogger
from llm_reasoning.training_data import SFTBatches
import math, time

RUN_MODE = 'inspect'  # inspect | smoke | production
APPROVED_EARLY_STOP = False  # nếu pretrain dừng theo thời gian, cần review trước SFT.
PRETRAIN_CHECKPOINT = Path('/kaggle/input/llm-reasoning-pretrain/latest.pt') if ON_KAGGLE else REPO_DIR/'artifacts/nb04_pretrain/latest.pt'
TOKENIZER = Path('/kaggle/input/llm-reasoning-data/pretrain/tokenizer.json') if ON_KAGGLE else REPO_DIR/'artifacts/data/pretrain/tokenizer.json'
SFT_ROOT = Path('/kaggle/input/llm-reasoning-data/sft_gsm8k') if ON_KAGGLE else REPO_DIR/'artifacts/data/sft_gsm8k'
OUTPUT_ROOT = Path('/kaggle/working/sft_run') if ON_KAGGLE else REPO_DIR/'artifacts/nb06_sft'
RESUME_FROM = None  # chỉ checkpoint SFT cùng dataset/config/thiết bị
MAX_EPOCHS_THIS_SESSION = 1
if RUN_MODE not in ('inspect', 'smoke', 'production'):
    raise ValueError('RUN_MODE không hợp lệ')
if RUN_MODE == 'production' and (not ON_KAGGLE or not torch.cuda.is_available() or DIRTY):
    raise RuntimeError('Production yêu cầu Kaggle CUDA, commit sạch')
if RESUME_FROM is not None and RUN_MODE != 'production':
    raise ValueError('Chỉ resume cùng SFT production run')
tokenizer = ByteBPETokenizer.load(TOKENIZER)
manifest = json.loads((SFT_ROOT/'manifest.json').read_text(encoding='utf-8'))
if manifest['sft_sources'] != ['gsm8k'] or manifest['tokenizer_sha256'] != file_sha256(TOKENIZER):
    raise ValueError('SFT phải là GSM8K-only và dùng đúng tokenizer pretrain')
_, provenance = load_model_artifact(PRETRAIN_CHECKPOINT, tokenizer_path=TOKENIZER)
if provenance['stage'] != 'pretrain':
    raise ValueError('SFT phải bắt đầu từ checkpoint pretrain')
if RUN_MODE == 'production' and provenance['checkpoint_kind'] != 'training':
    raise ValueError('Không dùng checkpoint smoke làm nguồn production SFT')
if RUN_MODE == 'production' and not APPROVED_EARLY_STOP and provenance['train_state']['attempted_steps'] < provenance['total_steps']:
    raise ValueError('Checkpoint pretrain chưa hoàn tất; chưa bắt đầu SFT')
print('Mode:', RUN_MODE, 'pretrain SHA:', file_sha256(PRETRAIN_CHECKPOINT),
      'SFT rows:', manifest['splits'])
'''),
    code(r'''
if RUN_MODE != 'inspect':
    cfg = make_config('smoke' if RUN_MODE == 'smoke' else 'target')
    train_cfg = cfg.train
    if RUN_MODE == 'smoke':
        train_cfg = replace(train_cfg, micro_batch_size=1, gradient_accumulation_steps=1,
                            sft_seq_len=manifest['max_seq_len'], precision='fp32')
    device = 'cpu' if RUN_MODE == 'smoke' else 'cuda'
    train_data = SFTBatches(SFT_ROOT, 'train', batch_size=train_cfg.micro_batch_size,
                             pad_token_id=tokenizer.special_to_id['<pad>'],
                             max_seq_len=train_cfg.sft_seq_len,
                             max_batches=4 if RUN_MODE == 'smoke' else None)
    stream = ResumableBatchStream(train_data, fingerprint=train_data.fingerprint,
                                  seed=cfg.data.seed, shuffle=True)
    total_steps = math.ceil(len(train_data)/train_cfg.gradient_accumulation_steps) * cfg.data.sft_max_epochs
    print('Microbatches/epoch:', len(train_data), 'planned max steps:', total_steps,
          'fingerprint:', train_data.fingerprint)
else:
    print('Inspect hoàn tất; chưa tạo optimizer và chưa train.')
'''),
    markdown('''## Tiếp tục và dừng theo validation

    Lần đầu nạp **chỉ model weights** từ pretrain; optimizer, scheduler, scaler SFT mới. Resume từ `latest.pt` SFT mới khôi phục đủ trạng thái. Bắt đầu 1 epoch; xem validation rồi quyết định có tiếp tục tối đa 3 epoch. Ngân sách 8M token được kiểm ở biên update nên có thể vượt tối đa một accumulation window. Giữ snapshot mỗi epoch để chọn checkpoint theo validation sau review; `latest.pt` phục vụ resume. Copy output vào nguồn lưu bền vững sau phiên.'''),
    code(r'''
if RUN_MODE != 'inspect':
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    latest = OUTPUT_ROOT/'latest.pt'
    log_dir = OUTPUT_ROOT/'logs'
    if RESUME_FROM is None:
        if latest.exists() or (log_dir/'metrics.csv').exists():
            raise FileExistsError('Output cũ tồn tại; chọn thư mục mới hoặc RESUME_FROM')
        torch.manual_seed(cfg.data.seed)
        model, _ = load_model_artifact(PRETRAIN_CHECKPOINT, tokenizer_path=TOKENIZER, device=device)
        trainer = Trainer(model, train_cfg, stage='sft', total_steps=total_steps)
    else:
        trainer = load_checkpoint(RESUME_FROM, stream, tokenizer_path=TOKENIZER,
                                  device=device, expected_stage='sft')
        if trainer.total_steps != total_steps or trainer.config != train_cfg:
            raise ValueError('Config SFT/total_steps thay đổi sau resume')
    kind = 'smoke' if RUN_MODE == 'smoke' else 'training'
    run_manifest = dict(stage='sft', kind=kind, code_commit=CODE_COMMIT,
                        pretrained_checkpoint_sha256=file_sha256(PRETRAIN_CHECKPOINT),
                        tokenizer_sha256=file_sha256(TOKENIZER),
                        data_fingerprint=train_data.fingerprint,
                        train_config=asdict(train_cfg), total_steps=total_steps, device=device)
    run_manifest_path = OUTPUT_ROOT/'run_manifest.json'
    if run_manifest_path.exists():
        if json.loads(run_manifest_path.read_text(encoding='utf-8')) != run_manifest:
            raise ValueError('Run manifest SFT thay đổi; không thể resume cùng output')
    else:
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2)+'\n', encoding='utf-8')
    with TrainingLogger(log_dir, resume_step=trainer.state.attempted_steps if RESUME_FROM else None) as logger:
        trainer.logger = logger
        epochs_completed = 0
        last_saved = time.monotonic()
        while epochs_completed < MAX_EPOCHS_THIS_SESSION and stream.epoch < cfg.data.sft_max_epochs \
                and trainer.state.attempted_steps < trainer.total_steps \
                and trainer.state.processed_tokens < cfg.data.sft_max_tokens:
            rows = trainer.train_epoch(stream, max_updates=1)
            if rows and time.monotonic() - last_saved >= train_cfg.checkpoint_interval_minutes * 60:
                save_checkpoint(latest, trainer, stream, tokenizer_path=TOKENIZER, checkpoint_kind=kind)
                last_saved = time.monotonic()
            if stream.cursor == len(stream.batches):
                validation = SFTBatches(SFT_ROOT, 'validation', batch_size=train_cfg.micro_batch_size,
                                        pad_token_id=tokenizer.special_to_id['<pad>'],
                                        max_seq_len=train_cfg.sft_seq_len,
                                        max_batches=2 if RUN_MODE == 'smoke' else None)
                score = trainer.evaluate(validation)
                snapshot = OUTPUT_ROOT/f'epoch_{stream.epoch + 1}.pt'
                save_checkpoint(snapshot, trainer, stream, tokenizer_path=TOKENIZER, checkpoint_kind=kind)
                print('Epoch', stream.epoch + 1, 'validation loss', score['loss'], 'snapshot', snapshot)
                epochs_completed += 1
                if stream.epoch + 1 < cfg.data.sft_max_epochs and trainer.state.processed_tokens < cfg.data.sft_max_tokens:
                    stream.next_epoch()
            elif not rows:
                raise RuntimeError('Stream chưa hết nhưng không có update; kiểm tra total_steps')
        save_checkpoint(latest, trainer, stream, tokenizer_path=TOKENIZER, checkpoint_kind=kind)
    print('SFT updates:', trainer.state.attempted_steps, 'tokens:', trainer.state.processed_tokens,
          'epoch:', stream.epoch, 'cursor:', stream.cursor, 'latest:', latest)
'''),
])

print('Built notebooks 03-06 in', DEST)
