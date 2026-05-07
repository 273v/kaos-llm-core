"""Minimal categorical Tree-structured Parzen Estimator (Bergstra 2011).

Phase 17.1: foundational search algorithm for ``MiproV2Optimizer``'s
joint (instruction x demos) candidate search. The full continuous-TPE
machinery (KDE kernels, magic clipping, gradient sampling) is unused
because MIPROv2's variables are categorical — discrete indices into
candidate lists. This module ships the categorical variant only,
~250 LOC, no scipy / numpy / optuna dependency.

Algorithm reference: Bergstra, Bardenet, Bengio, Kégl. *Algorithms
for Hyper-Parameter Optimization*. NeurIPS 2011. The categorical
formulation is mirrored from Optuna's
``_ParzenEstimator._calculate_categorical_distributions``.

Step-by-step (one ``suggest()`` call):

1. If we have fewer than ``n_startup_trials`` observations, return a
   uniform random draw — TPE has no signal yet.
2. Sort observations by score *descending* (we are maximizing).
3. Split at the top-``gamma`` quantile into ``good`` (top gamma fraction)
   and ``bad`` (everything else).
4. For each categorical variable, build two Laplace-smoothed
   categorical distributions over its K values:

       l(x = k) = (count_good(k) + prior) / (|good| + K · prior)
       g(x = k) = (count_bad(k)  + prior) / (|bad|  + K · prior)

5. Draw ``n_ei_candidates`` joint samples by sampling each variable
   independently from its ``l_d``. Score every candidate by the
   product ``∏_d l_d(c_d) / g_d(c_d)`` (Expected Improvement is
   monotonic in this ratio for TPE — see Bergstra 2011 §4).
6. Return the candidate with the highest ratio.

Why this design over Optuna:

* The KAOS optimizer catalog has zero algorithmic optional deps. Adding
  optuna would create two code paths (present / absent) and split the
  integration test gates. ~250 LOC of pure Python is the consistency-
  preserving choice.
* The categorical search space is trivially small (≤4 dims, ≤18 vals
  each in MIPROv2). Sophisticated TPE features (continuous KDE, magic
  clipping) are unused.
* The algorithm is well-specified and testable: ``test_tpe_search.py``
  asserts TPE beats pure random on a synthetic optimum-grid signal.
* Optuna's default ``n_startup_trials = 10`` starves small ``num_trials``
  runs (DSPy's auto=light gives 12 trials → 10 random + 2 TPE-guided).
  This implementation defaults to ``max(3, num_trials // 4)``, which
  gives 3 random + 9 TPE-guided in the same setting.

The :class:`CategoricalSearcher` protocol is the swap point: future
gaussian-process or optuna-backed searchers implement the same two
methods and drop in.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "CategoricalSearcher",
    "TPESearcher",
    "TPESearcherConfig",
]


# ---------------------------------------------------------------------------
# Protocol — the swappable searcher contract
# ---------------------------------------------------------------------------


@runtime_checkable
class CategoricalSearcher(Protocol):
    """Minimal protocol for categorical search algorithms.

    Implementations need exactly two methods plus the ``space``
    attribute. ``suggest`` returns a fresh point in the search space;
    ``observe`` records a (point, score) pair so future suggestions
    are informed by it.

    The ``full_eval`` flag on :meth:`observe` is for the *caller's*
    bookkeeping (e.g. to mark which observations came from a noisy
    minibatch eval vs. the full validation set). The TPE
    implementation in this module treats both kinds equally — the
    posterior is pooled — matching DSPy/Optuna behavior where a
    promoted full-eval is just another observation in the same
    distribution.
    """

    space: dict[str, int]
    """Mapping of categorical variable name → number of values (K)."""

    def suggest(self) -> dict[str, int]:
        """Return one fresh point in the search space."""
        ...

    def observe(self, params: dict[str, int], score: float, *, full_eval: bool) -> None:
        """Record an observation. ``score`` is to be maximized."""
        ...


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TPESearcherConfig:
    """Tunable hyperparameters for :class:`TPESearcher`.

    Defaults are chosen for the small-N MIPROv2 use case
    (``num_trials`` in 12-30, dimensions in 1-4, K per dim in 4-18).
    Most users should not touch any of these.
    """

    gamma: float = 0.25
    """Quantile threshold splitting good vs bad observations.

    The top ``gamma`` fraction of observations (by score) are the
    'good' set; the rest are 'bad'. Bergstra 2011 uses 0.25 for small
    sample sizes; Optuna defaults to ``min(⌈0.1·n⌉, 25)/n`` which is
    too conservative for our N≈12-20 trial counts.
    """

    n_startup_trials: int | None = None
    """Number of initial uniform-random trials before TPE engages.

    If ``None``, the searcher chooses ``max(3, num_trials // 4)`` at
    construction time when ``num_trials`` is supplied — otherwise it
    defaults to 3. Optuna uses 10 unconditionally, which starves
    small runs.
    """

    n_ei_candidates: int = 24
    """Number of candidate points sampled per ``suggest()`` call.

    The candidate with the highest ``l/g`` ratio wins. 24 matches
    Optuna's default.
    """

    prior_weight: float = 1.0
    """Laplace smoothing weight for the categorical distributions.

    Without smoothing, a category with zero good-set observations
    has ``l(k) = 0`` and would never be sampled. ``prior_weight=1.0``
    is uniform additive smoothing — equivalent to a flat Beta prior.
    """

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma < 1.0:
            msg = f"TPESearcherConfig.gamma must be in (0, 1); got {self.gamma}"
            raise ValueError(msg)
        if self.n_startup_trials is not None and self.n_startup_trials < 1:
            msg = (
                "TPESearcherConfig.n_startup_trials must be >= 1 when set; "
                f"got {self.n_startup_trials}"
            )
            raise ValueError(msg)
        if self.n_ei_candidates < 1:
            msg = f"TPESearcherConfig.n_ei_candidates must be >= 1; got {self.n_ei_candidates}"
            raise ValueError(msg)
        if self.prior_weight < 0.0:
            msg = f"TPESearcherConfig.prior_weight must be >= 0; got {self.prior_weight}"
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# TPESearcher
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Observation:
    params: dict[str, int]
    score: float
    full_eval: bool


class TPESearcher:
    """Categorical Tree-structured Parzen Estimator.

    Parameters
    ----------
    space:
        Mapping of variable name → number of categories (K). Every
        suggestion is a dict with the same keys, with each value an
        integer in ``range(K)``.
    num_trials:
        Optional total trial count, used to derive ``n_startup_trials``
        when the config does not pin it. Pass it if known so the
        startup phase scales with the budget.
    config:
        Optional :class:`TPESearcherConfig`. Defaults are chosen for
        small-N runs (≤20 trials).
    seed:
        Random seed for reproducibility.

    Example
    -------
    >>> tpe = TPESearcher(space={"a": 4, "b": 4}, num_trials=20, seed=0)
    >>> point = tpe.suggest()
    >>> tpe.observe(point, score=0.42, full_eval=False)

    The class implements the :class:`CategoricalSearcher` protocol.
    """

    __slots__ = ("_observations", "_rng", "config", "n_startup_trials", "space")

    def __init__(
        self,
        *,
        space: dict[str, int],
        num_trials: int | None = None,
        config: TPESearcherConfig | None = None,
        seed: int = 0,
    ) -> None:
        if not space:
            msg = "TPESearcher.space must contain at least one variable"
            raise ValueError(msg)
        for name, k in space.items():
            if k < 1:
                msg = f"TPESearcher.space[{name!r}] must be >= 1 categories; got {k}"
                raise ValueError(msg)

        self.space: dict[str, int] = dict(space)
        self.config = config or TPESearcherConfig()
        # Derive n_startup_trials lazily so callers can pass num_trials
        # at construction without re-instantiating the config.
        if self.config.n_startup_trials is not None:
            self.n_startup_trials = self.config.n_startup_trials
        elif num_trials is not None:
            self.n_startup_trials = max(3, num_trials // 4)
        else:
            self.n_startup_trials = 3

        self._rng = random.Random(seed)
        self._observations: list[_Observation] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def observations(self) -> list[_Observation]:
        """Read-only snapshot of recorded observations."""
        return list(self._observations)

    @property
    def n_observations(self) -> int:
        return len(self._observations)

    def suggest(self) -> dict[str, int]:
        """Return one fresh point in the search space.

        Cold-start (``n_observations < n_startup_trials``): uniform
        random across every variable. Steady state: TPE-guided draw
        per the algorithm in the module docstring.
        """
        if self.n_observations < self.n_startup_trials:
            return self._uniform_random()
        return self._tpe_suggest()

    def observe(self, params: dict[str, int], score: float, *, full_eval: bool = False) -> None:
        """Record an observation. ``score`` is to be maximized.

        ``full_eval`` is informational — TPE pools both kinds of
        observations into the same posterior, matching Optuna /
        DSPy semantics on promoted trials.
        """
        self._validate_point(params)
        self._observations.append(
            _Observation(params=dict(params), score=float(score), full_eval=full_eval)
        )

    # ------------------------------------------------------------------
    # Internal: random and TPE-guided sampling
    # ------------------------------------------------------------------

    def _uniform_random(self) -> dict[str, int]:
        return {name: self._rng.randrange(k) for name, k in self.space.items()}

    def _tpe_suggest(self) -> dict[str, int]:
        good, bad = self._split_good_bad()
        # Edge case: gamma rounds to len(observations), so bad is empty.
        # Fall back to uniform random rather than divide-by-zero on g(x).
        if not good or not bad:
            return self._uniform_random()

        # Per-variable Laplace-smoothed distributions, computed once
        # and reused across all n_ei_candidates draws.
        l_per_var: dict[str, list[float]] = {}
        g_per_var: dict[str, list[float]] = {}
        for name, k in self.space.items():
            l_per_var[name] = self._categorical_density(good, name, k)
            g_per_var[name] = self._categorical_density(bad, name, k)

        # Sample n_ei candidates from l, score by ∏ l/g, return argmax.
        best_candidate: dict[str, int] | None = None
        best_ratio = -math.inf
        # Use log-ratios for numerical stability with multiple dims.
        for _ in range(self.config.n_ei_candidates):
            candidate: dict[str, int] = {}
            log_l = 0.0
            log_g = 0.0
            for name, k in self.space.items():
                weights = l_per_var[name]
                drawn = self._weighted_choice(weights, k)
                candidate[name] = drawn
                log_l += math.log(max(weights[drawn], 1e-300))
                log_g += math.log(max(g_per_var[name][drawn], 1e-300))
            log_ratio = log_l - log_g
            if log_ratio > best_ratio:
                best_ratio = log_ratio
                best_candidate = candidate

        # best_candidate is non-None because n_ei_candidates >= 1
        # (validated in TPESearcherConfig.__post_init__).
        assert best_candidate is not None
        return best_candidate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_good_bad(self) -> tuple[list[_Observation], list[_Observation]]:
        sorted_obs = sorted(self._observations, key=lambda o: o.score, reverse=True)
        n = len(sorted_obs)
        n_good = max(1, math.ceil(self.config.gamma * n))
        n_good = min(n_good, n - 1)  # Always leave at least one in bad.
        return sorted_obs[:n_good], sorted_obs[n_good:]

    def _categorical_density(
        self, observations: list[_Observation], name: str, k: int
    ) -> list[float]:
        prior = self.config.prior_weight
        counts = [0] * k
        for obs in observations:
            counts[obs.params[name]] += 1
        denom = len(observations) + k * prior
        return [(c + prior) / denom for c in counts]

    def _weighted_choice(self, weights: list[float], k: int) -> int:
        # random.Random.choices is O(k) per call but we only sample one
        # value at a time, so the constant factor is fine.
        return self._rng.choices(range(k), weights=weights, k=1)[0]

    def _validate_point(self, params: dict[str, int]) -> None:
        if set(params.keys()) != set(self.space.keys()):
            missing = set(self.space) - set(params)
            extra = set(params) - set(self.space)
            msg = (
                "TPESearcher.observe: params keys must match the search space; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
            raise ValueError(msg)
        for name, value in params.items():
            k = self.space[name]
            if not isinstance(value, int) or not 0 <= value < k:
                msg = f"TPESearcher.observe: params[{name!r}]={value!r} must be an int in [0, {k})"
                raise ValueError(msg)
