# Counting Islands

Counts the number of islands in an `M x N` binary grid, where `1` represents
land and `0` represents ocean.

## Assumptions

- **Connectivity is 4-directional only** (up, down, left, right). Diagonal
  neighbors are **not** considered connected. For example:

  ```
  1 0
  0 1
  ```

  is treated as **2 separate islands**, not one.

- The grid must contain only `0` and `1`. Any other value raises a
  `ValueError`.
- Every row must have exactly `N` columns, matching the declared dimensions.
  A mismatch raises a `ValueError`.
- An empty grid (`M = 0`, or rows with zero columns) is valid input and
  returns `0`.

## Algorithm

Iterative **BFS** (breadth-first search) using a `collections.deque` as the
queue:

1. Scan every cell of the grid in row-major order.
2. When an unvisited land cell (`1`) is found, run BFS from it. BFS explores
   all 4-directionally reachable land cells and marks them visited — this
   marks the entire island in one pass.
3. Each completed BFS run corresponds to exactly one island, so increment
   the island counter once per BFS call.
4. Continue scanning; cells already marked visited are skipped.

Iterative BFS was chosen over recursive DFS specifically to avoid Python's
recursion limit, which could be hit on a single large connected island in a
big grid (e.g. one island spanning thousands of cells in a long snake shape).

### Complexity

- **Time: `O(M * N)`** — every cell is enqueued and processed at most once,
  regardless of how the islands are shaped.
- **Space: `O(M * N)`** — the `visited` boolean matrix is the dominant cost;
  the BFS queue holds at most `O(M * N)` cells in the worst case (e.g. a
  single grid that is entirely land).

## Project structure

```
task_1/
├── islands.py        # algorithm + CLI entry point
├── test_islands.py    # pytest test suite
└── README.md
```

## Usage

### Run from the command line

Input format: first line is `M N`, followed by `M` lines each containing `N`
space-separated values (`0` or `1`).

```bash
python3 islands.py
```

Example session (typed or piped via stdin):

```
3 4
0 0 0 1
0 0 1 1
0 1 0 1
```

Output:

```
2
```

You can also pipe input directly:

```bash
printf "3 4\n0 0 0 1\n0 0 1 1\n0 1 0 1\n" | python3 islands.py
```

### Use as a library

```python
from islands import count_islands

grid = [
    [0, 1, 0],
    [0, 0, 0],
    [0, 1, 1],
]
print(count_islands(grid))  # 2
```

`count_islands` derives `M` and `N` directly from the matrix, so it cannot be
called with dimensions that disagree with the actual data — there is no
separate `M`/`N` argument to get out of sync.

## Running tests

```bash
pip install pytest
pytest test_islands.py -v
```

The suite covers:

- All three provided examples.
- All-zeros and all-ones grids.
- Single-cell grids (land and ocean).
- Diagonal and anti-diagonal placements (confirming they are **not**
  connected).
- Single-row and single-column islands.
- Empty grid / empty-row grid.
- Input validation: wrong row count, wrong row length, invalid cell values.
