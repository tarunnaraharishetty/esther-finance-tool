"""Financial Analyzer scoring layer (Phase 3+).

Currently exposes the technical sub-score system:

* :class:`~src.intelligence.analyzer.technical.TechnicalScores` — the
  full set of normalized [0, 100] sub-scores and the combiner outputs.
* :func:`~src.intelligence.analyzer.technical.score_technicals` —
  builds a :class:`TechnicalScores` instance from an OHLCV DataFrame.

Later phases will add valuation, AI-explanation, and DB-persistence
modules in this package.
"""

from src.intelligence.analyzer.calibration import (
    PAIRINGS as CALIBRATION_PAIRINGS,
)
from src.intelligence.analyzer.calibration import (
    CalibrationReading,
    lookup_readings,
)
from src.intelligence.analyzer.explanation import (
    CITATION_TAGS,
    AnalyzerExplanation,
    AnalyzerInputs,
    CitationValidationError,
    build_grounded_explanation,
    extract_citations,
    validate_citations,
)
from src.intelligence.analyzer.scenarios import (
    ScenarioModel,
    build_scenarios,
)
from src.intelligence.analyzer.technical import (
    TechnicalScores,
    TechnicalSubscores,
    score_technicals,
)
from src.intelligence.analyzer.valuation import (
    ValuationEnsemble,
    ValuationEstimate,
    build_valuation,
)

__all__ = [
    "CALIBRATION_PAIRINGS",
    "CITATION_TAGS",
    "AnalyzerExplanation",
    "AnalyzerInputs",
    "CalibrationReading",
    "CitationValidationError",
    "ScenarioModel",
    "TechnicalScores",
    "TechnicalSubscores",
    "ValuationEnsemble",
    "ValuationEstimate",
    "build_grounded_explanation",
    "build_scenarios",
    "build_valuation",
    "extract_citations",
    "lookup_readings",
    "score_technicals",
    "validate_citations",
]
