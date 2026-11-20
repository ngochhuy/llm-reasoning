"""Read-only checks before the single production Kaggle training run."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from llm_reasoning.checkpoint import file_sha256
from llm_reasoning.tokenizer import ByteBPETokenizer


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(['git', '-C', str(repo), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def inspect(data_root: Path, output_root: Path, repo: Path, *,
            production: bool = True, expected_commit: str | None = None,
            quota_confirmed: bool = False) -> dict:
    """Return evidence and explicit pass/fail/pending checks; never start training."""
    checks = []

    def check(name: str, ok: bool | None, detail) -> None:
        checks.append(dict(name=name, status='pending' if ok is None else 'pass' if ok else 'fail', detail=detail))

    commit = _git(repo, 'rev-parse', 'HEAD')
    dirty = _git(repo, 'status', '--porcelain')
    ahead = _git(repo, 'rev-list', '--count', 'origin/main..HEAD')
    check('git_commit', commit is not None and (expected_commit is None or commit == expected_commit),
          dict(actual=commit, expected=expected_commit))
    check('git_clean', dirty == '', dict(dirty=dirty, ahead_of_origin_main=ahead))
    if production:
        check('commit_pinned', expected_commit is not None and len(expected_commit) == 40,
              'Pass the reviewed 40-character commit SHA')
        # The notebook clones this SHA on Kaggle; local ahead count shows whether it has been pushed.
        check('commit_pushed', ahead is not None and ahead.isdigit() and int(ahead) == 0,
              dict(ahead_of_origin_main=int(ahead) if ahead and ahead.isdigit() else None))

    kaggle = Path('/kaggle/working').is_dir()
    gpu = torch.cuda.is_available()
    gpu_info = dict(torch=str(torch.__version__), cuda_build=torch.version.cuda,
                    cuda_available=gpu, name=torch.cuda.get_device_name(0) if gpu else None,
                    total_vram_bytes=torch.cuda.get_device_properties(0).total_memory if gpu else None)
    if production:
        check('kaggle_runtime', kaggle, dict(kaggle=kaggle))
        check('cuda_gpu', gpu, gpu_info)
        check('kaggle_gpu_quota', True if quota_confirmed else None,
              'Confirm remaining GPU hours and session limit in your Kaggle account')

    pretrain = data_root / 'pretrain'
    sft = data_root / 'sft_gsm8k'
    try:
        pre_manifest = json.loads((pretrain/'manifest.json').read_text(encoding='utf-8'))
        sft_manifest = json.loads((sft/'manifest.json').read_text(encoding='utf-8'))
        tokenizer_path = pretrain/'tokenizer.json'
        tokenizer = ByteBPETokenizer.load(tokenizer_path)
        sha = file_sha256(tokenizer_path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        check('data_manifests', False, str(exc))
        return dict(checks=checks, ready=False, code_commit=commit,
                    data_root=str(data_root), output_root=str(output_root))

    check('tokenizer_hash', sha == pre_manifest.get('tokenizer_sha256')
          == sft_manifest.get('tokenizer_sha256') and tokenizer.vocab_size == pre_manifest.get('vocab_size'),
          dict(actual_sha256=sha, pretrain_sha256=pre_manifest.get('tokenizer_sha256'),
               sft_sha256=sft_manifest.get('tokenizer_sha256'), vocab_size=tokenizer.vocab_size))
    block = pre_manifest.get('block_size')
    check('pretrain_format', block == 512 and pre_manifest.get('token_dtype') == 'uint16'
          and pre_manifest.get('byte_order') == sys.byteorder,
          dict(block_size=block, token_dtype=pre_manifest.get('token_dtype'),
               byte_order=pre_manifest.get('byte_order'), host_byte_order=sys.byteorder))

    for source, target in (('fineweb', 80_000_000), ('math', 20_000_000)):
        try:
            source_manifest = pre_manifest['sources'][source]
            for split in ('train', 'validation'):
                info = source_manifest['shards'][split]
                paths = [pretrain/source/split/name for name in info['files']]
                actual = sum(path.stat().st_size for path in paths) // 2
                valid = bool(paths) and all(path.is_file() and path.stat().st_size % (block*2) == 0
                                            for path in paths) and actual == info['written_tokens']
                if split == 'train':
                    valid = valid and actual >= target - block
                else:
                    valid = valid and actual > 0
                check(f'{source}_{split}_shards', valid,
                      dict(files=len(paths), written_tokens=info['written_tokens'],
                           actual_tokens=actual, target_tokens=target if split == 'train' else None))
        except (KeyError, OSError, TypeError, ValueError) as exc:
            check(f'{source}_shards', False, str(exc))

    split = sft_manifest.get('gsm_split', {})
    only_gsm = sft_manifest.get('sft_sources') == ['gsm8k'] and not (sft/'orca_train.jsonl').exists()
    check('sft_source_and_split', only_gsm and split.get('validation') == 500
          and split.get('test') == 1319 and sft_manifest.get('test_examples') == 1319,
          dict(sources=sft_manifest.get('sft_sources'), validation=split.get('validation'),
               test=split.get('test'), legacy_orca_present=(sft/'orca_train.jsonl').exists()))
    for label, name in (('train', 'gsm_train.jsonl'), ('validation', 'gsm_validation.jsonl'),
                        ('test', 'gsm_test.jsonl')):
        path = sft/name
        try:
            with path.open(encoding='utf-8') as file:
                count = sum(1 for line in file if line.strip())
            expected = (sft_manifest['splits'][f'gsm_{label}']['accepted']
                        if label != 'test' else sft_manifest['test_examples'])
            check(f'sft_{label}_rows', count == expected and count > 0,
                  dict(actual=count, expected=expected, file=str(path)))
        except (KeyError, OSError, ValueError) as exc:
            check(f'sft_{label}_rows', False, str(exc))

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_root, prefix='.preflight_', delete=True):
            pass
        free = shutil.disk_usage(output_root).free
        check('output_writable_and_space', free >= 2 * 1024**3,
              dict(path=str(output_root), free_bytes=free, minimum_bytes=2 * 1024**3))
    except OSError as exc:
        check('output_writable_and_space', False, str(exc))

    return dict(checks=checks, ready=all(c['status'] == 'pass' for c in checks),
                code_commit=commit, data_root=str(data_root), output_root=str(output_root),
                gpu=gpu_info)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--expected-commit')
    parser.add_argument('--quota-confirmed', action='store_true',
                        help='Use only after personally checking Kaggle GPU quota')
    parser.add_argument('--local', action='store_true', help='Check data without requiring Kaggle/CUDA')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    result = inspect(args.data_root, args.output_root, args.repo,
                     production=not args.local, expected_commit=args.expected_commit,
                     quota_confirmed=args.quota_confirmed)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding='utf-8')
    print(encoded)
    if not result['ready']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
