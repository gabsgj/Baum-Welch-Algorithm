"""Tests for hmm_core.utils.normalization."""

import numpy as np
import pytest

from hmm_core.utils.normalization import normalize_rows, normalize_vector


class TestNormalizeVector:
    """Unit tests for normalize_vector."""

    def test_basic(self):
        v = np.array([2.0, 3.0, 5.0])
        result = normalize_vector(v)
        assert result.shape == (3,)
        assert abs(result.sum() - 1.0) < 1e-12

    def test_already_normalized(self):
        v = np.array([0.25, 0.25, 0.5])
        result = normalize_vector(v)
        np.testing.assert_allclose(result.sum(), 1.0)

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="zero"):
            normalize_vector(np.array([0.0, 0.0, 0.0]))

    def test_single_element(self):
        result = normalize_vector(np.array([7.0]))
        np.testing.assert_allclose(result, [1.0])


class TestNormalizeRows:
    """Unit tests for normalize_rows."""

    def test_basic(self):
        m = np.array([[1.0, 2.0], [3.0, 1.0]])
        result = normalize_rows(m)
        np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0])

    def test_already_stochastic(self):
        m = np.array([[0.3, 0.7], [0.6, 0.4]])
        result = normalize_rows(m)
        np.testing.assert_allclose(result, m, atol=1e-12)

    def test_zero_row_raises(self):
        m = np.array([[1.0, 2.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="zero"):
            normalize_rows(m)

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            normalize_rows(np.array([1.0, 2.0]))
