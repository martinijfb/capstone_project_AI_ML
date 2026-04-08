"""
Add weekly results from the portal to the .npy data files.

Usage:
    python src/add_results.py <week_number>

Example:
    python src/add_results.py 1

Reads from:
    data/weekly_inputs/w{week}_inputs.txt
    data/weekly_outputs/w{week}_outputs.txt

Appends to:
    data/function_{N}/initial_inputs.npy
    data/function_{N}/initial_outputs.npy
"""

import sys
import numpy as np
from numpy import array, float64

INITIAL_COUNTS = {1: 10, 2: 10, 3: 15, 4: 30, 5: 20, 6: 20, 7: 30, 8: 40}


def parse_text_file(filepath, week):
    """Parse the portal's text file format.

    The file may have entries that span multiple lines (e.g. long arrays wrap).
    Each entry starts with a digit (the week number) or a '['.
    We find the entry for the given week by looking for balanced brackets.
    """
    with open(filepath) as f:
        content = f.read()

    # Split into entries: each starts at a line beginning with a digit followed by tab,
    # or we just split by top-level list patterns
    # Strategy: find all top-level [...] blocks
    entries = []
    depth = 0
    start = None
    for i, ch in enumerate(content):
        if ch == '[' and depth == 0:
            start = i
        if ch == '[':
            depth += 1
        if ch == ']':
            depth -= 1
            if depth == 0 and start is not None:
                entries.append(content[start:i+1])
                start = None

    if week > len(entries):
        raise ValueError(f"Week {week} not found in {filepath} (only {len(entries)} entries)")

    data_str = entries[week - 1]
    return eval(data_str)


def main(week: int):
    inputs_path = f'data/weekly_inputs/w{week}_inputs.txt'
    outputs_path = f'data/weekly_outputs/w{week}_outputs.txt'

    # Parse text files
    try:
        new_inputs = parse_text_file(inputs_path, week)
        new_outputs = parse_text_file(outputs_path, week)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"Make sure the files exist at {inputs_path} and {outputs_path}")
        sys.exit(1)

    assert len(new_inputs) == 8, f"Expected 8 inputs, got {len(new_inputs)}"
    assert len(new_outputs) == 8, f"Expected 8 outputs, got {len(new_outputs)}"

    # Check expected point count to detect double-append
    expected_points = {n: INITIAL_COUNTS[n] + week for n in range(1, 9)}

    print(f"Adding week {week} results")
    print("=" * 75)
    print(f"{'Func':>5} {'Before':>8} {'After':>8} {'New X (first 3)':>20} {'New Y':>12} {'Status':>10}")
    print("-" * 75)

    for n in range(1, 9):
        X = np.load(f'data/function_{n}/initial_inputs.npy')
        Y = np.load(f'data/function_{n}/initial_outputs.npy')

        current_count = len(Y)

        # Double-append check
        if current_count >= expected_points[n]:
            print(f"  F{n} {current_count:>8} {'—':>8} {'—':>20} {'—':>12} {'SKIPPED':>10}")
            print(f"       Already has {current_count} points (expected {expected_points[n]}). Already appended?")
            continue

        if current_count != expected_points[n] - 1:
            print(f"  F{n} {current_count:>8} {'—':>8} {'—':>20} {'—':>12} {'WARNING':>10}")
            print(f"       Expected {expected_points[n] - 1} points before append, got {current_count}.")
            print(f"       Proceeding anyway...")

        new_x = np.array(new_inputs[n - 1])
        new_y = float(new_outputs[n - 1])

        X_new = np.vstack([X, new_x.reshape(1, -1)])
        Y_new = np.append(Y, new_y)

        np.save(f'data/function_{n}/initial_inputs.npy', X_new)
        np.save(f'data/function_{n}/initial_outputs.npy', Y_new)

        # Verify
        X_check = np.load(f'data/function_{n}/initial_inputs.npy')
        Y_check = np.load(f'data/function_{n}/initial_outputs.npy')
        assert X_check.shape[0] == current_count + 1
        assert len(Y_check) == current_count + 1

        x_preview = ', '.join(f'{v:.3f}' for v in new_x[:3])
        if len(new_x) > 3:
            x_preview += '...'
        print(f"  F{n} {current_count:>8} {current_count+1:>8} [{x_preview:>17}] {new_y:>12.4f} {'OK':>10}")

    # Summary
    print("\n" + "=" * 75)
    print("Verification — current state:")
    print(f"{'Func':>5} {'Points':>8} {'Dims':>6} {'Best Y':>12} {'Last Y':>12} {'New Best?':>10}")
    print("-" * 60)
    for n in range(1, 9):
        X = np.load(f'data/function_{n}/initial_inputs.npy')
        Y = np.load(f'data/function_{n}/initial_outputs.npy')
        best_idx = np.argmax(Y)
        is_new = best_idx == len(Y) - 1
        print(f"  F{n} {len(Y):>8} {X.shape[1]:>6} {Y.max():>12.4f} {Y[-1]:>12.4f} {'★' if is_new else '':>10}")

    print(f"\n✓ Week {week} results added successfully.")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python src/add_results.py <week_number>")
        print("Example: python src/add_results.py 1")
        sys.exit(1)

    week = int(sys.argv[1])
    main(week)
