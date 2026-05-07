"""Tier 2.5: regression tests for the per-optimizer Config dataclasses.

Verifies the three construction patterns introduced in Phase 16.5
work for every optimizer that has a Config:

  1. Typed config object: ``Optimizer(metric, config=Cfg(field=value))``
  2. Backwards-compatible kwargs: ``Optimizer(metric, field=value)``
  3. Config + overrides: ``Optimizer(metric, config=cfg, field=override)``

Also verifies that:

  - ``__post_init__`` validation fires for invalid field values
  - Unknown override keys raise ``TypeError`` (typo protection)
  - Both construction patterns produce instances with identical
    field values
"""

from __future__ import annotations

import pytest

from kaos_llm_core.optimization import (
    BootstrapConfig,
    BootstrapOptimizer,
    CodecOptimizer,
    CodecOptimizerConfig,
    CoOptimizer,
    CoOptimizerConfig,
    HyperparameterConfig,
    HyperparameterOptimizer,
    InstructionConfig,
    InstructionOptimizer,
    MiproLiteConfig,
    MiproLiteOptimizer,
    ModelOptimizer,
    ModelOptimizerConfig,
    ReflectiveConfig,
    ReflectiveOptimizer,
    resolve_config,
)


def _metric(_pred, _gold) -> float:
    return 1.0


# ---------------------------------------------------------------------------
# resolve_config helper
# ---------------------------------------------------------------------------


class TestResolveConfig:
    def test_no_config_no_overrides_returns_default(self) -> None:
        cfg = resolve_config(BootstrapConfig, None, {})
        assert cfg.max_examples == 5
        assert cfg.score_threshold == 1.0

    def test_no_config_with_overrides_constructs(self) -> None:
        cfg = resolve_config(BootstrapConfig, None, {"max_examples": 10})
        assert cfg.max_examples == 10

    def test_config_no_overrides_returns_unchanged(self) -> None:
        original = BootstrapConfig(max_examples=7)
        cfg = resolve_config(BootstrapConfig, original, {})
        assert cfg is original

    def test_config_with_overrides_replaces(self) -> None:
        original = BootstrapConfig(max_examples=7)
        cfg = resolve_config(BootstrapConfig, original, {"score_threshold": 0.5})
        assert cfg.max_examples == 7
        assert cfg.score_threshold == 0.5
        # Replace produces a new instance, not a mutation
        assert original.score_threshold == 1.0

    def test_unknown_override_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="bogus_field"):
            resolve_config(BootstrapConfig, None, {"bogus_field": 42})


# ---------------------------------------------------------------------------
# Per-optimizer construction patterns
# ---------------------------------------------------------------------------


class TestBootstrapPatterns:
    def test_kwargs_pattern(self) -> None:
        opt = BootstrapOptimizer(metric=_metric, max_examples=8)
        assert opt.max_examples == 8
        assert isinstance(opt.config, BootstrapConfig)

    def test_config_pattern(self) -> None:
        opt = BootstrapOptimizer(metric=_metric, config=BootstrapConfig(max_examples=8))
        assert opt.max_examples == 8

    def test_config_plus_override(self) -> None:
        opt = BootstrapOptimizer(
            metric=_metric,
            config=BootstrapConfig(max_examples=8),
            score_threshold=0.5,
        )
        assert opt.max_examples == 8
        assert opt.score_threshold == 0.5

    def test_invalid_field_value_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="max_examples must be >= 1"):
            BootstrapOptimizer(metric=_metric, max_examples=0)


class TestInstructionPatterns:
    def test_kwargs_pattern(self) -> None:
        opt = InstructionOptimizer(metric=_metric, max_trials=5, patience=3)
        assert opt.max_trials == 5
        assert opt.patience == 3

    def test_config_pattern(self) -> None:
        opt = InstructionOptimizer(
            metric=_metric, config=InstructionConfig(max_trials=5, patience=3)
        )
        assert opt.max_trials == 5
        assert opt.patience == 3

    def test_invalid_max_trials_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_trials"):
            InstructionOptimizer(metric=_metric, max_trials=0)


class TestHyperparameterPatterns:
    def test_kwargs_pattern_search_space_positional(self) -> None:
        opt = HyperparameterOptimizer(
            _metric, {"temperature": [0.1, 0.5]}, strategy="random", max_trials=10
        )
        assert opt.search_space == {"temperature": [0.1, 0.5]}
        assert opt.strategy == "random"

    def test_config_pattern(self) -> None:
        opt = HyperparameterOptimizer(
            metric=_metric,
            config=HyperparameterConfig(search_space={"temperature": [0.0, 1.0]}, strategy="grid"),
        )
        assert opt.strategy == "grid"

    def test_invalid_strategy_rejected(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            HyperparameterOptimizer(
                metric=_metric,
                search_space={"x": [1]},
                strategy="bogus",
            )


class TestCoOptimizerPatterns:
    def test_kwargs_pattern(self) -> None:
        opt = CoOptimizer(metric=_metric, max_bootstrap_examples=2)
        assert opt.max_bootstrap_examples == 2

    def test_config_pattern(self) -> None:
        opt = CoOptimizer(
            metric=_metric,
            config=CoOptimizerConfig(max_bootstrap_examples=2),
        )
        assert opt.max_bootstrap_examples == 2


class TestMiproLitePatterns:
    def test_kwargs_pattern(self) -> None:
        opt = MiproLiteOptimizer(metric=_metric, n_trials=12, minibatch_size=10)
        assert opt.n_trials == 12
        assert opt.minibatch_size == 10

    def test_config_pattern(self) -> None:
        opt = MiproLiteOptimizer(
            metric=_metric, config=MiproLiteConfig(n_trials=12, minibatch_size=10)
        )
        assert opt.n_trials == 12
        assert opt.minibatch_size == 10

    def test_unknown_override_rejected(self) -> None:
        with pytest.raises(TypeError, match="bogus"):
            MiproLiteOptimizer(metric=_metric, bogus=42)


class TestCodecOptimizerPatterns:
    def test_kwargs_pattern_default_codecs(self) -> None:
        from kaos_llm_core.codecs import JSONCodec

        opt = CodecOptimizer(metric=_metric)
        assert JSONCodec in opt.codecs

    def test_config_pattern(self) -> None:
        from kaos_llm_core.codecs import JSONCodec

        opt = CodecOptimizer(metric=_metric, config=CodecOptimizerConfig(codecs=[JSONCodec]))
        assert opt.codecs == [JSONCodec]


class TestModelOptimizerPatterns:
    def test_positional_models_works(self) -> None:
        opt = ModelOptimizer(_metric, ["openai:gpt-4"], min_score=0.6)
        assert opt.models == ["openai:gpt-4"]
        assert opt.min_score == 0.6

    def test_config_pattern(self) -> None:
        opt = ModelOptimizer(
            metric=_metric,
            config=ModelOptimizerConfig(models=["openai:gpt-4"], min_score=0.6),
        )
        assert opt.models == ["openai:gpt-4"]

    def test_empty_models_rejected(self) -> None:
        with pytest.raises(ValueError, match="models"):
            ModelOptimizer(metric=_metric, models=[])


class TestReflectivePatterns:
    def test_kwargs_pattern(self) -> None:
        opt = ReflectiveOptimizer(critic_model="anthropic:claude-haiku-4-5", max_trials=2)
        assert opt.critic_model == "anthropic:claude-haiku-4-5"
        assert opt.max_trials == 2

    def test_config_pattern(self) -> None:
        opt = ReflectiveOptimizer(
            config=ReflectiveConfig(critic_model="anthropic:claude-haiku-4-5", max_trials=2)
        )
        assert opt.critic_model == "anthropic:claude-haiku-4-5"
        assert opt.max_trials == 2

    def test_proposer_defaults_to_critic(self) -> None:
        opt = ReflectiveOptimizer(critic_model="anthropic:claude-haiku-4-5")
        assert opt.proposer_model == "anthropic:claude-haiku-4-5"
