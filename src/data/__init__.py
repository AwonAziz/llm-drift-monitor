"""Data layer: dataset ingestion, reference construction, production shift simulation."""

from .datasets import (  # noqa: F401
    DatasetInfo,
    balanced_holdout,
    build_reference,
    intent_catalog,
    load_banking77,
    write_dataset,
    write_manifest,
)
from .shifts import (  # noqa: F401
    Regime,
    TrafficSimulator,
    TrafficWindow,
    default_schedule,
)