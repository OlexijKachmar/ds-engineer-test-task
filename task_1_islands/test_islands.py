import pytest

from islands import count_islands, validate_matrix


class TestCountIslandsProvidedExamples:
    def test_example_1(self):
        matrix = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 1],
        ]
        assert count_islands(matrix) == 2

    def test_example_2(self):
        matrix = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ]
        assert count_islands(matrix) == 3

    def test_example_3(self):
        matrix = [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
        ]
        assert count_islands(matrix) == 2


class TestCountIslandsEdgeCases:
    def test_all_zeros(self):
        matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        assert count_islands(matrix) == 0

    def test_all_ones(self):
        matrix = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        assert count_islands(matrix) == 1

    def test_single_cell_ocean(self):
        assert count_islands([[0]]) == 0

    def test_single_cell_island(self):
        assert count_islands([[1]]) == 1

    def test_diagonal_ones_are_separate_islands(self):
        matrix = [
            [1, 0],
            [0, 1],
        ]
        assert count_islands(matrix) == 2

    def test_anti_diagonal_ones_are_separate_islands(self):
        matrix = [
            [0, 1],
            [1, 0],
        ]
        assert count_islands(matrix) == 2

    def test_one_horizontal_island(self):
        matrix = [
            [1, 1, 1, 1],
        ]
        assert count_islands(matrix) == 1

    def test_one_vertical_island(self):
        matrix = [
            [1],
            [1],
            [1],
            [1],
        ]
        assert count_islands(matrix) == 1

    def test_empty_matrix(self):
        assert count_islands([]) == 0

    def test_matrix_with_empty_rows(self):
        assert count_islands([[]]) == 0


class TestValidateMatrix:
    def test_valid_matrix_passes(self):
        matrix = [
            [0, 1, 0],
            [0, 0, 1],
        ]
        # Should not raise
        validate_matrix(matrix, M=2, N=3)

    def test_invalid_row_count_raises(self):
        matrix = [
            [0, 1, 0],
        ]
        with pytest.raises(ValueError, match="Expected 2 rows"):
            validate_matrix(matrix, M=2, N=3)

    def test_invalid_row_length_raises(self):
        matrix = [
            [0, 1, 0],
            [0, 1],  # wrong length
        ]
        with pytest.raises(ValueError, match="Expected 3 columns"):
            validate_matrix(matrix, M=2, N=3)

    def test_invalid_cell_value_raises(self):
        matrix = [
            [0, 2, 0],
            [0, 1, 1],
        ]
        with pytest.raises(ValueError, match="Invalid value"):
            validate_matrix(matrix, M=2, N=3)

    def test_negative_cell_value_raises(self):
        matrix = [
            [0, -1, 0],
            [0, 1, 1],
        ]
        with pytest.raises(ValueError, match="Invalid value"):
            validate_matrix(matrix, M=2, N=3)
