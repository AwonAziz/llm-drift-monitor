"""Output-quality monitoring: calibration, delayed labels, label-free proxies."""

from .outputs import (  # noqa: F401
    OutputQualityMonitor,
    QualityResult,
    adaptive_ece,
    expected_calibration_error,
    maximum_calibration_error,
    score_health,
    severity_from,
    snapshot_baseline,
    worst,
)
