/**
 * Provenance map type — shared by every grounded-AI surface that
 * carries per-token source attribution.
 *
 * Keys are the verbatim numeric tokens that appeared in the prose
 * (e.g. ``"$180.50"``, ``"73%"``). Values are short dotted source-
 * field paths the UI renders as tooltips (e.g. ``"row.last_price"``
 * on /research, ``"left.trust_score.score"`` on /compare/narrative).
 *
 * Empty when no tokens grounded or the validator hasn't run. The
 * `ProvenanceProse` component reads this map directly — surfaces
 * that produce a ProvenanceMap can render hover-traceable prose
 * without any per-surface adapter code.
 */
export type ProvenanceMap = Record<string, string>;
