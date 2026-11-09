"""Numbers check: every number in the Writer's prose must appear in the evidence.

Evidence is what the pipeline actually holds: query results, findings, verifier corrections and the
tables the code rendered. A number "appears" if it matches an evidence value after the rounding
the Writer used, or after the unit changes analysts make (MW -> GW, share -> %, days -> years).

This catches invented or mistyped numbers. It cannot catch a correct number attached to the wrong
label; rendering tables in code (visuals.py) removes the main place that happens.
"""

import re

# 1,234,567.89 | 1234.5 | .5 ; optional leading $ or minus.
_NUMBER = re.compile(r"(?<![\w.])[-−]?\$?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?![\w])")
_MARKER = re.compile(r"<!--gq:\w+:\d+-->")
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_LIST_ORDINAL = re.compile(r"^\s*\d+\.\s", re.MULTILINE)

# How an evidence value can legitimately reappear in prose.
_SCALES = (1, 1 / 1000, 1000, 100, 1 / 100, 1 / 365.25, 1 / 12)


def _parse(raw: str) -> tuple[float, int]:
    text = raw.replace(",", "")
    decimals = len(text.split(".")[1]) if "." in text else 0
    return float(text), decimals


def numbers_in(text: str) -> list[tuple[str, float, int]]:
    """(as written, value, decimals) for each number in prose, skipping years, list ordinals,
    visual markers and markdown tables (tables are rendered by code from the database)."""
    text = _MARKER.sub(" ", text)
    text = _TABLE_LINE.sub(" ", text)
    text = _LIST_ORDINAL.sub(" ", text)
    out = []
    for m in _NUMBER.finditer(text):
        if re.search(r"[A-Za-z][-‐–]$", text[max(0, m.start() - 2):m.start() + (m.group(0)[0] in "-−")]):
            continue  # part of a name like EIA-860
        raw = m.group(1)
        value, decimals = _parse(raw)
        if decimals == 0 and "," not in raw and 1990 <= value <= 2045:
            continue  # a year
        out.append((raw, value, decimals))
    return out


def evidence_values(texts: list[str]) -> set[float]:
    values = set()
    for t in texts:
        for m in _NUMBER.finditer(t):
            values.add(_parse(m.group(1))[0])
    return values


def _matches(value: float, decimals: int, evidence: set[float]) -> bool:
    tol = 0.5 * 10 ** -decimals + 1e-9
    # A small whole number ("2 times", "6 regions") matches almost anything after a unit change,
    # so only allow it as a rounded value or a share written as a percent.
    scales = (1, 100) if decimals == 0 and abs(value) < 100 else _SCALES
    for e in evidence:
        for k in scales:
            if abs(e * k - value) <= tol:
                return True
    return False


def unsupported_numbers(prose: str, evidence_texts: list[str]) -> list[str]:
    """Numbers in the prose that no evidence value supports, as written, without repeats."""
    evidence = evidence_values(evidence_texts)
    bad = [raw for raw, value, decimals in numbers_in(prose) if not _matches(value, decimals, evidence)]
    return list(dict.fromkeys(bad))


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z*(])")
_BULLET = re.compile(r"^(\s*[-*]\s+(?:\*\*[^*]+\*\*:?\s*)?)(.*)$")


def drop_unsupported_sentences(prose: str, bad: list[str]) -> tuple[str, int]:
    """Last resort after a failed revision: remove sentences holding an unsupported number, except in
    the opening answer (which is left as is and flagged). A bullet left empty is removed. Returns the
    new prose and how many sentences were dropped."""
    if not bad:
        return prose, 0
    pattern = re.compile(r"(?<![\w.,])(?:" + "|".join(re.escape(b) for b in bad) + r")(?![\w]|[.,]\d)")
    paragraphs = prose.split("\n\n")
    dropped = 0
    for p_i in range(1, len(paragraphs)):
        lines = paragraphs[p_i].split("\n")
        kept_lines = []
        for line in lines:
            if not pattern.search(line) or _TABLE_LINE.match(line) or _MARKER.search(line):
                kept_lines.append(line)
                continue
            m = _BULLET.match(line)
            prefix, body = (m.group(1), m.group(2)) if m else ("", line)
            sentences = _SENTENCE.split(body)
            keep = [x for x in sentences if not pattern.search(x)]
            dropped += len(sentences) - len(keep)
            if keep:
                kept_lines.append(prefix + " ".join(keep))
        paragraphs[p_i] = "\n".join(kept_lines)
    return "\n\n".join(x for x in paragraphs if x.strip()), dropped
