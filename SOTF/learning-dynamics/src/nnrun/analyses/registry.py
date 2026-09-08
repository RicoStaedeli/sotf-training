from __future__ import annotations

from typing import Any, Dict, Callable

from .base import AnalysisResult
from .acc_curves import run_acc_curves
from .ranking_flips import run_ranking_flips
from .clustering import run_clustering
from .late_bloomers import run_late_bloomers, run_early_faders
from .delta_to_best import run_delta_to_best
from .hyper_parameter import run_hyper_analysis
from .weight_statistics import run_weight_analysis
from .weight_rank_correspondence import run_weight_rank_correspondence
from .weight_early_signal import run_weight_early_signal

ANALYSIS_REGISTRY: dict[str, Callable[[Any, Dict[str, Any],Dict[str, Any]], AnalysisResult]] = {
    "acc_curves": run_acc_curves,
    "ranking_flips": run_ranking_flips,
    "clustering": run_clustering,
    "late_bloomers": run_late_bloomers,
    "early_faders": run_early_faders,
    "delta_to_best": run_delta_to_best,
    "hyper_analysis": run_hyper_analysis,
    "weight_analysis": run_weight_analysis,
    "weight_rank_correspondence": run_weight_rank_correspondence,
    "weight_early_signal": run_weight_early_signal,
}
