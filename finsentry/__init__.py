"""FinSentry — financial transaction anomaly & AML detection engine.

A transaction-monitoring style pipeline that combines an unsupervised ML
ensemble, an AML typology rule engine, and money-flow graph analysis into a
single explainable risk score per account, ranked into an alert queue.
"""

__version__ = "1.0.0"

from .pipeline import run_pipeline  # noqa: F401
from .datagen import generate_dataset, REPORTING_THRESHOLD  # noqa: F401
