from collections import deque
from typing import List, Tuple, Dict

from collections import deque
from typing import List, Tuple


def bfs(start_pos: Tuple[int, int], visited: List[List[bool]], island_matrix: List[List[int]]) -> None:
    M = len(island_matrix)
    N = len(island_matrix[0])

    queue = deque()
    visited[start_pos[0]][start_pos[1]] = True
    queue.append(start_pos)

    while queue:
        curr_i, curr_j = queue.popleft()
        neighbours = [
            (curr_i - 1, curr_j), (curr_i + 1, curr_j),
            (curr_i, curr_j - 1), (curr_i, curr_j + 1)
        ]
        for neighbour in neighbours:
            neighbor_i, neighbor_j = neighbour
            matrix_fit_condition = (neighbor_i >= 0 and neighbor_i < M) and (neighbor_j >= 0 and neighbor_j < N)

            if matrix_fit_condition:
                is_island = (island_matrix[neighbor_i][neighbor_j] == 1)
                if is_island and not visited[neighbor_i][neighbor_j]:
                    visited[neighbor_i][neighbor_j] = True
                    queue.append(neighbour)


def count_islands(island_matrix: List[List[int]]) -> int:
    """
    Counts the number of islands in a 2D binary grid using BFS.
    Islands are connected horizontally/vertically (4-directional), not diagonally.

    M and N are derived from the matrix itself, so the function cannot be called
    with dimensions that disagree with the actual data.

    Time complexity: O(M*N) - every cell is visited at most once.
    Space complexity: O(M*N) - visited matrix + BFS queue in the worst case (all 1s).
    """
    if not island_matrix or not island_matrix[0]:
        return 0

    M = len(island_matrix)
    N = len(island_matrix[0])

    visited = [[False] * N for _ in range(M)]
    islands_count = 0
    for i in range(M):
        for j in range(N):
            is_island = (island_matrix[i][j] == 1)
            if not visited[i][j] and is_island:
                bfs((i, j), visited, island_matrix)
                islands_count += 1

    return islands_count


def validate_matrix(island_matrix: List[List[int]], M: int, N: int) -> None:
    """
    Validates that the matrix matches the declared dimensions and contains
    only 0s and 1s. Raises ValueError with a clear message otherwise.
    """
    if len(island_matrix) != M:
        raise ValueError(f"Expected {M} rows, got {len(island_matrix)}")

    for i, row in enumerate(island_matrix):
        if len(row) != N:
            raise ValueError(f"Expected {N} columns in row {i}, got {len(row)}: {row}")
        for j, value in enumerate(row):
            if value not in (0, 1):
                raise ValueError(f"Invalid value {value} at ({i}, {j}); only 0 or 1 allowed")


def read_grid_from_stdin() -> List[List[int]]:
    M, N = map(int, input().split())

    island_matrix = []
    for _ in range(M):
        row = list(map(int, input().split()))
        island_matrix.append(row)

    validate_matrix(island_matrix, M, N)
    return island_matrix


if __name__ == "__main__":
    island_matrix = read_grid_from_stdin()
    print(count_islands(island_matrix))
