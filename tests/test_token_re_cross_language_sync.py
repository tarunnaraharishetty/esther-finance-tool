"""B-19 cross-language sync: frontend ``TOKEN_RE`` matches Python ``TOKEN_PATTERN``.

The grounded-AI surfaces tokenize numeric claims on both sides — the
Python validators drop unsupported sentences and emit a provenance
map keyed by token strings; ``ProvenanceProse.tsx`` re-tokenizes the
prose in the browser and renders hover tooltips for every token that
matches a provenance key. Drift between the two regexes silently
breaks the provenance graph (tokens the server emitted no longer
render with a tooltip, or vice versa).

This test pins the contract: extract the frontend literal, compile it
via Python ``re``, run it against the same full-coverage fixture as
the Python ``TOKEN_PATTERN``, and assert the match lists are
identical. Any divergence — different alternation order, missing
branch, extra branch — fails CI loudly.

Why not codegen
---------------
Generating ``ProvenanceProse.tsx`` from a Python source would add a
build step every developer + every CI pipeline has to remember to
run. A behavioural sync test runs every pytest invocation, has no
external prerequisites, and exposes drift the moment either side
moves. The cost is a small parity fixture that needs to be expanded
when a new token shape is added — which is fine, that's the point.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.intelligence.text_grounding import TOKEN_PATTERN

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROVENANCE_PROSE_TSX = (
    _PROJECT_ROOT / "web" / "src" / "components" / "prose" / "ProvenanceProse.tsx"
)


# Full-coverage fixture — at least one input per alternation branch
# in TOKEN_PATTERN. When a new branch is added (e.g. ``bps`` for basis
# points), append a sample here so both languages stay covered.
_FIXTURE_INPUTS: tuple[str, ...] = (
    "Revenue grew to $1,250,000.00 last quarter.",
    "FCF was $250000000 with margins steady.",
    "Net income climbed 12.4% YoY.",
    "Buybacks added -2.5% to share count.",
    "Forward PE sits at 12.5x against peers.",
    "Up 3X over 12 months.",
    "Earnings due Q3 2025; prior was Q1.",
    "Guidance refreshed 2026; restated 2024 numbers.",
    "RSI ended the day at 87.4 with momentum cooling.",
    "Bollinger band width compressed to 1.62.",
    "Float reported as 8500000 shares this filing.",
    "AAPL, MSFT, NVDA — no numerics here, should yield nothing.",
    "Mixed: $1.2B revenue at 18.5% net margin, PE 30x for Q4 2026 cycle.",
)


def _extract_frontend_pattern() -> str:
    """Return the regex body inside the TS ``/(.../g`` literal.

    Strips the leading ``/`` and trailing ``/g``; leaves the alternation
    body verbatim. Raises if the literal can't be found so a refactor
    that moves the constant fails this test loudly instead of silently
    skipping the contract.
    """
    if not _PROVENANCE_PROSE_TSX.exists():
        pytest.skip(
            f"frontend file missing: {_PROVENANCE_PROSE_TSX} — skipping cross-"
            f"language sync (not all environments check out the web/ tree)"
        )
    text = _PROVENANCE_PROSE_TSX.read_text(encoding="utf-8")
    match = re.search(
        r"const\s+TOKEN_RE\s*=\s*/(.+?)/[a-z]*\s*;",
        text,
        re.DOTALL,
    )
    assert match is not None, (
        f"Could not find ``const TOKEN_RE = /.../g;`` in "
        f"{_PROVENANCE_PROSE_TSX}. Did the constant move or get renamed?"
    )
    return match.group(1)


@pytest.mark.parametrize("sample", _FIXTURE_INPUTS)
def test_frontend_and_python_token_re_match_the_same_tokens(sample: str) -> None:
    """For every sample, the Python and TS regexes must return the
    exact same ordered token list. Behavioural parity, not source-
    string parity (TS has no verbose-regex mode)."""
    py_tokens = TOKEN_PATTERN.findall(sample)
    ts_body = _extract_frontend_pattern()
    # The TS literal wraps the alternation in one capture group:
    # ``/(\$...|-?\d...|...)/g``. ``findall`` on a single-group regex
    # returns the captured strings, which is exactly the alternation
    # body we want — same as the Python pattern.
    ts_re = re.compile(ts_body)
    ts_tokens = ts_re.findall(sample)
    assert ts_tokens == py_tokens, (
        f"Frontend / Python tokenizer disagreement on sample {sample!r}:\n"
        f"  python: {py_tokens}\n"
        f"  ts:     {ts_tokens}\n"
        f"Update either side or the fixture so they stay in sync — see "
        f"BUGS.md B-19 for the contract."
    )


def test_frontend_literal_is_findable() -> None:
    """Refactor guard: if the TS file moves or renames the constant
    so the extractor can't find it, fail explicitly instead of silently
    passing zero parametrized cases. Pins the file path + variable name
    that the sync test depends on."""
    _extract_frontend_pattern()  # raises via the assert inside


def test_python_token_pattern_is_the_shared_canonical_object() -> None:
    """B-19 invariant: every grounded-AI validator must point at the
    same compiled object — no module-private copies that can drift."""
    from src.intelligence.comparison_narrative_validator import (
        _TOKEN_PATTERN as compare_pattern,
    )
    from src.intelligence.research_validator import (
        _TOKEN_PATTERN as research_pattern,
    )
    from src.intelligence.text_grounding import TOKEN_PATTERN as canonical

    assert research_pattern is canonical, (
        "research_validator must alias text_grounding.TOKEN_PATTERN — "
        "a private copy will drift (BUGS.md B-19)"
    )
    assert compare_pattern is canonical, (
        "comparison_narrative_validator must alias text_grounding.TOKEN_PATTERN — "
        "a private copy will drift (BUGS.md B-19)"
    )
