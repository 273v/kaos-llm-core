"""Tests for Latin Hypercube Sampling in HyperparameterOptimizer."""

from __future__ import annotations

import random

from kaos_llm_core.optimization.hyperparameter import _latin_hypercube_samples


class TestLatinHypercube:
    def test_returns_n_samples(self) -> None:
        rng = random.Random(42)
        samples = _latin_hypercube_samples(
            {"temperature": [0.0, 0.3, 0.7, 1.0]}, n_samples=4, rng=rng
        )
        assert len(samples) == 4

    def test_stratification_coverage(self) -> None:
        """With n_samples == len(values), every value must appear exactly once."""
        rng = random.Random(42)
        values = [0.0, 0.3, 0.7, 1.0]
        samples = _latin_hypercube_samples({"temperature": values}, n_samples=4, rng=rng)
        drawn = sorted(s["temperature"] for s in samples)
        assert drawn == sorted(values)

    def test_reproducible_with_seed(self) -> None:
        rng_a = random.Random(7)
        rng_b = random.Random(7)
        a = _latin_hypercube_samples(
            {"temperature": [0.0, 0.5, 1.0], "top_p": [0.1, 0.5, 0.9]},
            n_samples=3,
            rng=rng_a,
        )
        b = _latin_hypercube_samples(
            {"temperature": [0.0, 0.5, 1.0], "top_p": [0.1, 0.5, 0.9]},
            n_samples=3,
            rng=rng_b,
        )
        assert a == b

    def test_oversampling_with_replacement(self) -> None:
        """n_samples > len(values): values are reused cyclically."""
        rng = random.Random(0)
        samples = _latin_hypercube_samples({"temperature": [0.0, 1.0]}, n_samples=5, rng=rng)
        assert len(samples) == 5
        drawn = [s["temperature"] for s in samples]
        # Each value should appear roughly evenly: 3 of one, 2 of the other.
        assert set(drawn) == {0.0, 1.0}

    def test_empty_grid(self) -> None:
        rng = random.Random(0)
        assert _latin_hypercube_samples({}, n_samples=5, rng=rng) == []

    def test_zero_samples(self) -> None:
        rng = random.Random(0)
        assert _latin_hypercube_samples({"t": [0.0]}, n_samples=0, rng=rng) == []

    def test_multi_param_all_columns_present(self) -> None:
        rng = random.Random(3)
        samples = _latin_hypercube_samples(
            {"a": [1, 2, 3], "b": ["x", "y", "z"]}, n_samples=3, rng=rng
        )
        assert len(samples) == 3
        for s in samples:
            assert set(s.keys()) == {"a", "b"}
