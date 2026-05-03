"""Strip Intel MKL warnings from a Jupyter notebook's saved cell outputs.

Usage:
    python3 src/clean_notebook.py notebooks/week_06.ipynb

Background: anaconda Python on this Mac is x86_64 (running through Rosetta on
the M-series chip), so Intel MKL emits a deprecation warning every time it
loads. The warnings have no effect on correctness — they just clutter the
notebook. This script filters them out of saved stream outputs.
"""
import json
import re
import sys


def clean(path: str) -> int:
    nb = json.load(open(path))
    mkl_re = re.compile(r'^Intel MKL WARNING.*$')

    dropped = 0
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        kept_outputs = []
        for o in c.get('outputs', []):
            if o.get('output_type') == 'stream' and 'text' in o:
                t = o['text']
                if isinstance(t, list):
                    t = ''.join(t)
                lines = t.split('\n')
                filtered = []
                for line in lines:
                    if mkl_re.match(line.strip()) or 'oneAPI Math Kernel Library' in line:
                        dropped += 1
                        continue
                    filtered.append(line)
                new_text = '\n'.join(filtered)
                if new_text.strip() == '':
                    continue
                o['text'] = new_text
            kept_outputs.append(o)
        c['outputs'] = kept_outputs

    json.dump(nb, open(path, 'w'), indent=1)
    return dropped


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 src/clean_notebook.py <notebook.ipynb>')
        sys.exit(1)
    n = clean(sys.argv[1])
    print(f'Stripped {n} MKL warning lines from {sys.argv[1]}')
