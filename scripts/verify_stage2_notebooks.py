"""Execute notebook code cells in a fresh Python process; optional tiny smoke flow."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def notebook_script(path: Path, smoke: bool) -> str:
    notebook = json.loads(path.read_text(encoding='utf-8'))
    sources = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            compile(source, str(path), 'exec')
            if smoke:
                source = source.replace("RUN_MODE = 'inspect'", "RUN_MODE = 'smoke'")
            sources.append(source)
    return '\n\n'.join(sources) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Run notebooks 04-06 with tiny CPU updates')
    parser.add_argument('--sanity', action='store_true', help='Also run notebook 03 tests and toy checks')
    args = parser.parse_args()
    names = ['03_model_sanity_checks.ipynb', '04_pretraining.ipynb',
             '05_kv_cache_profiling.ipynb', '06_math_sft.ipynb']
    with tempfile.TemporaryDirectory() as directory:
        for name in names:
            source = notebook_script(ROOT/'notebooks'/name, args.smoke and name != names[0])
            print('Compiled:', name)
            if (name == names[0] and not args.sanity) or (name != names[0] and not args.smoke):
                continue
            for artifact in ('nb04_pretrain', 'nb05_profile', 'nb06_sft'):
                source = source.replace("REPO_DIR/'artifacts/" + artifact + "'",
                                        'Path(' + repr(str(Path(directory)/artifact)) + ')')
            source = source.replace("REPO_DIR/'artifacts/nb04_pretrain/latest.pt'",
                                    'Path(' + repr(str(Path(directory)/'nb04_pretrain'/'latest.pt')) + ')')
            path = Path(directory)/f'{name}.py'
            path.write_text(source, encoding='utf-8')
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'
            subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env, check=True)
    print('Notebook verification complete')


if __name__ == '__main__':
    main()
