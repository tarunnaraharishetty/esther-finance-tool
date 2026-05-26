"""Tiny FIFO-eviction cap for ``dict`` containers.

Several long-running components mirror LLM output into dicts
(``brief_cache``, ``opp_brief_cache``) that grow with every
``Sparkles``-style refresh. Without a cap the persisted JSON snapshot
balloons over weeks until load / save latency starts dominating the
tick budget.

This module exposes one helper: :func:`bounded_set`. It sets the key
in place and, if the dict has overflowed its cap, drops the oldest
entry. The eviction is FIFO (oldest insertion), not strict LRU — Python
dicts preserve insertion order so this is one ``next(iter(d))`` away,
and the use case ("don't grow forever") doesn't need recency-tracking
machinery.

If a key already exists, we delete-then-set so re-writes refresh the
key's position in the iteration order — the closest thing to LRU
semantics a plain dict supports.
"""

from __future__ import annotations


def bounded_set[K, V](
    container: dict[K, V],
    key: K,
    value: V,
    *,
    max_entries: int,
) -> None:
    """Set ``container[key] = value`` and evict the oldest entry if over cap.

    Args:
        container: The dict to update. Mutated in place.
        key: Key to set / refresh.
        value: Value to associate.
        max_entries: Maximum entries to retain. Must be >= 1; values
            below this raise ``ValueError`` because a zero-cap dict is
            a footgun (every set immediately evicts the just-added key).

    A re-set of an existing key moves it to the end of the iteration
    order (delete-then-set), which is the closest thing to LRU
    semantics a plain dict supports — so a hot key won't be evicted
    just because new keys keep landing.
    """
    if max_entries < 1:
        raise ValueError(f"max_entries must be >= 1, got {max_entries!r}")
    container.pop(key, None)
    container[key] = value
    # Evict from the front until we're at cap. Usually one pop.
    while len(container) > max_entries:
        oldest = next(iter(container))
        del container[oldest]


def bounded_trim[K, V](container: dict[K, V], *, max_entries: int) -> int:
    """Trim ``container`` from the front down to ``max_entries``.

    Returns the number of entries dropped. Useful on the hydrate path —
    when a snapshot loads a dict whose size exceeds the current cap,
    trim once instead of letting the next N sets each evict one entry.
    """
    if max_entries < 1:
        raise ValueError(f"max_entries must be >= 1, got {max_entries!r}")
    dropped = 0
    while len(container) > max_entries:
        oldest = next(iter(container))
        del container[oldest]
        dropped += 1
    return dropped


__all__ = ["bounded_set", "bounded_trim"]
