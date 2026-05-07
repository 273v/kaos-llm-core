"""Optimization — self-improving LLM programs."""

from kaos_llm_core.optimization.base import resolve_config
from kaos_llm_core.optimization.bootstrap import BootstrapConfig, BootstrapOptimizer
from kaos_llm_core.optimization.budget import Budget, BudgetTracker, StopReason
from kaos_llm_core.optimization.co_optimizer import CoOptimizer, CoOptimizerConfig
from kaos_llm_core.optimization.codec_optimizer import (
    CodecOptimizationResult,
    CodecOptimizer,
    CodecOptimizerConfig,
)
from kaos_llm_core.optimization.evaluation import EvalResult, evaluate
from kaos_llm_core.optimization.hyperparameter import (
    HyperparameterConfig,
    HyperparameterOptimizer,
)
from kaos_llm_core.optimization.instruction import (
    InstructionConfig,
    InstructionOptimizer,
)
from kaos_llm_core.optimization.mipro_lite import (
    MiproLiteConfig,
    MiproLiteOptimizer,
    MiproLiteResult,
)
from kaos_llm_core.optimization.mipro_v2 import (
    MiproV2BestConfig,
    MiproV2Config,
    MiproV2Optimizer,
    MiproV2Result,
)
from kaos_llm_core.optimization.model_optimizer import (
    ModelOptimizationResult,
    ModelOptimizer,
    ModelOptimizerConfig,
)
from kaos_llm_core.optimization.mutations import Mutation, MutationLog
from kaos_llm_core.optimization.pareto import (
    ParetoOptimizer,
    ParetoResult,
    compute_pareto_frontier,
)
from kaos_llm_core.optimization.recipes import (
    CostAwareModelSelector,
    ExampleFirstTuner,
    FriendlyPromptTuner,
)
from kaos_llm_core.optimization.reflective import (
    ReflectiveConfig,
    ReflectiveOptimizer,
)

__all__ = [
    "BootstrapConfig",
    "BootstrapOptimizer",
    "Budget",
    "BudgetTracker",
    "CoOptimizer",
    "CoOptimizerConfig",
    "CodecOptimizationResult",
    "CodecOptimizer",
    "CodecOptimizerConfig",
    "CostAwareModelSelector",
    "EvalResult",
    "ExampleFirstTuner",
    "FriendlyPromptTuner",
    "HyperparameterConfig",
    "HyperparameterOptimizer",
    "InstructionConfig",
    "InstructionOptimizer",
    "MiproLiteConfig",
    "MiproLiteOptimizer",
    "MiproLiteResult",
    "MiproV2BestConfig",
    "MiproV2Config",
    "MiproV2Optimizer",
    "MiproV2Result",
    "ModelOptimizationResult",
    "ModelOptimizer",
    "ModelOptimizerConfig",
    "Mutation",
    "MutationLog",
    "ParetoOptimizer",
    "ParetoResult",
    "ReflectiveConfig",
    "ReflectiveOptimizer",
    "StopReason",
    "compute_pareto_frontier",
    "evaluate",
    "resolve_config",
]
