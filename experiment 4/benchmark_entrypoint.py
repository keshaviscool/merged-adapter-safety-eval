"""Run the shared benchmark with experiment-local seed handling and provenance."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import runpy
import sys

from merge_seed import install_merge_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--provenance', required=True)
    args = parser.parse_args()
    provenance = json.loads(Path(args.provenance).read_text())
    install_merge_seed(provenance['merge_seed'])
    import yaml

    config = yaml.safe_load(Path(args.config).read_text())
    output_root = Path(config['output']['root'])
    sys.argv = ['run_benchmark.py', '--config', args.config]
    runpy.run_path(str(Path(__file__).parent / 'run_benchmark.py'), run_name='__main__')
    completed = list(output_root.glob('*/summary.json'))
    if len(completed) != 1:
        raise RuntimeError('Expected exactly one completed result in the unique run directory')
    provenance['packages'] = {p: version(p) for p in ('torch', 'transformers', 'peft', 'numpy')}
    (completed[0].parent / 'experiment4_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
