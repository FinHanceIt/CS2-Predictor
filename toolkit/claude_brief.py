"""
Claude layer — optional natural-language briefing + Q&A over the slate.

This is the honest meaning of "connected to Claude": the app calls the Anthropic
API to (a) write a short Romanian briefing over the deterministic predictions and
their Transurfing labels, and (b) answer free-text questions about the slate.

It is fully optional. If ANTHROPIC_API_KEY is missing or the `anthropic` package
is not installed, `available()` returns False and the app falls back to the
deterministic engine with a friendly note. Nothing here changes any probability —
Claude only explains numbers that already exist.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
_SYSTEM = (
    "Ești analistul intern al unui sistem de predicție CS2. Scrii în română, "
    "concis și direct, fără diacritice greșite. Nu inventezi cote sau cifre — "
    "folosești doar datele primite. Nu promiți câștiguri sigure; reamintești "
    "că modelul e ~58% cu varianță mare. Vorbești în termenii sistemului: "
    "RIDE = banda calmă fiabilă, CARE = favorit greu de tratat cu atenție, "
    "FLAT = aproape de monedă, AVOID = supra-încredere pedepsită empiric."
)


def available() -> tuple[bool, str]:
    """Return (is_available, reason). Reason explains what's missing if not."""
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return False, "Lipsește ANTHROPIC_API_KEY din .env."
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False, "Pachetul `anthropic` nu e instalat (pip install anthropic)."
    return True, "ok"


def _client():
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "").strip())


def _slate_lines(reads) -> str:
    """Compact text encoding of TransurfingReading objects for the prompt."""
    lines = []
    for x in reads:
        pend = "pendul" if getattr(x, "is_pendulum", False) else "-"
        lines.append(
            f"- {x.verdict_key.upper()} | {x.pick} {x.pick_conf*100:.0f}% "
            f"vs {x.team_b if x.pick == x.team_a else x.team_a} | "
            f"banda {x.band_label} (fiabilitate {x.band_hit_rate*100:.0f}%) | {pend}"
        )
    return "\n".join(lines)


def briefing(reads, results_summary: dict | None = None, model: str | None = None) -> str:
    """Generate a short Romanian briefing over the slate. Raises on API error."""
    ok, reason = available()
    if not ok:
        return f"(Stratul Claude indisponibil: {reason})"

    perf = ""
    if results_summary:
        perf = (
            f"\nPerformanță istorică: {results_summary.get('n','?')} meciuri, "
            f"acuratețe {results_summary.get('acc',0)*100:.0f}%, "
            f"Brier {results_summary.get('brier',0):.3f}."
        )
    counts = {}
    for x in reads:
        counts[x.verdict_key] = counts.get(x.verdict_key, 0) + 1
    header = (
        f"Slate de {len(reads)} meciuri. "
        f"RIDE={counts.get('ride',0)}, CARE={counts.get('caution',0)}, "
        f"FLAT={counts.get('neutral',0)}, AVOID={counts.get('avoid',0)}.{perf}\n\n"
    )
    prompt = (
        header
        + "Meciuri:\n" + _slate_lines(reads)
        + "\n\nScrie un briefing de 4-6 fraze: ce merită urmărit (RIDE), "
        "ce să eviți (AVOID), și o notă de disciplină. Fără liste lungi."
    )
    msg = _client().messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=600,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def ask(question: str, reads, model: str | None = None) -> str:
    """Answer a free-text question about the current slate. Raises on API error."""
    ok, reason = available()
    if not ok:
        return f"(Stratul Claude indisponibil: {reason})"
    context = "Slate curent:\n" + _slate_lines(reads)
    msg = _client().messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=600,
        system=_SYSTEM,
        messages=[{"role": "user", "content": f"{context}\n\nÎntrebare: {question}"}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
