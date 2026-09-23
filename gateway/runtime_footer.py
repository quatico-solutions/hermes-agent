"""Gateway runtime-metadata footer (model · context % · cwd), off by default to keep replies
minimal. Config: ``display.runtime_footer: {enabled: bool, fields: [model, context_pct, cwd]}``
(order shown; drop any to hide), per-platform override ``display.platforms.<p>.runtime_footer``,
toggled by ``/footer on|off``. Fields: ``model`` (vendor prefix dropped), ``context_pct`` (last-call
occupancy), ``latency`` (turn wall-clock, opt-in — NOT in the default set so an unset ``fields``
renders exactly as before), ``served_model`` (opt-in, ``alias → served``: the deployment a routing
proxy reported via ``x-litellm-model-id`` / ``x-litellm-model-api-base``, or Hermes' own fallback
route; skipped when the served model is the requested one), ``cwd`` (home-relative), and the
opt-in ledger fields ``calls``, ``tokens``, ``cost`` (whole-session totals from ``session_model_usage``).
``template`` (e.g. Slack ``"↳ _{footer}_"``) wraps the line in platform markup. ``gateway/run.py`` appends the footer to the
final response only (never to tool-progress or streaming partials); when streaming already
delivered the text, it goes out as a trailing message via ``send_trailing_footer()``."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "cwd")
_SEP = " · "


def _home_relative_cwd(cwd: str) -> str:
    """Return *cwd* with ``$HOME`` collapsed to ``~``.  Empty string if unset."""
    if not cwd:
        return ""
    try:
        home = os.path.expanduser("~")
        p = os.path.abspath(cwd)
        if home and (p == home or p.startswith(home + os.sep)):
            return "~" + p[len(home):]
        return p
    except Exception:
        return cwd


def _model_short(model: Optional[str]) -> str:
    """Drop ``vendor/`` prefix (``openai/gpt-5.4`` → ``gpt-5.4``)."""
    return model.rsplit("/", 1)[-1] if model else ""


def _env_cwd() -> str:
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        return os.environ.get("TERMINAL_CWD", "")
    return terminal_env("TERMINAL_CWD", "")


def resolve_footer_config(user_config: dict[str, Any] | None, platform_key: str | None = None) -> dict[str, Any]:
    """Resolve effective footer config: defaults (enabled=False) <
    ``display.runtime_footer`` < ``display.platforms.<platform_key>.runtime_footer``."""
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS), "template": ""}
    cfg = (user_config or {}).get("display") or {}
    plat_cfg = (cfg.get("platforms") or {}).get(platform_key) if platform_key else None
    sections = [cfg.get("runtime_footer"), plat_cfg.get("runtime_footer") if isinstance(plat_cfg, dict) else None]
    for section in sections:
        if not isinstance(section, dict):
            continue
        if "enabled" in section:
            resolved["enabled"] = bool(section.get("enabled"))
        if isinstance(section.get("fields"), list) and section["fields"]:
            resolved["fields"] = [str(f) for f in section["fields"]]
        if isinstance(section.get("template"), str):
            resolved["template"] = section["template"]
    return resolved


def _format_latency(seconds: float) -> str:
    """Humanize a turn duration: ``<1s``, ``22s``, ``1m05s``."""
    if seconds < 1:
        return "<1s"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    m, sec = divmod(total, 60)
    return f"{m}m{sec:02d}s"


_LEDGER_FIELDS = frozenset({"calls", "tokens", "cost"})


def _human(n: int) -> str:
    return f"{n / 1_000_000:.1f}M" if n >= 1_000_000 else f"{n / 1_000:.1f}k" if n >= 1_000 else str(n)


def session_ledger(session_id: Optional[str], db_path: Optional[str]) -> Optional[dict]:
    """Whole-session usage totals from ``session_model_usage``, or None on any failure.

    Reads the LEDGER, not the agent's ``session_*`` counters: those start at 0 when the agent object
    is created and are never restored, so after a gateway restart or cache eviction they cover only
    "since this process picked the session up". The ledger spans every task (main loop, compression,
    title generation, background review) and is flushed at turn finalize, before the footer is built.
    Still a FLOOR: a background review spawned after the reply lands in the next turn's footer.
    Read-only URI connection — the gateway owns the writer; WAL permits readers."""
    if not session_id or not db_path:
        return None
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
        try:
            row = con.execute(
                "SELECT COALESCE(SUM(api_call_count),0),"
                " COALESCE(SUM(input_tokens + cache_read_tokens + cache_write_tokens),0),"
                " COALESCE(SUM(output_tokens),0), COALESCE(SUM(estimated_cost_usd),0)"
                " FROM session_model_usage WHERE session_id = ?", (session_id,)).fetchone()
        finally:
            con.close()
    except Exception:
        return None
    if not row or not row[0]:
        return None
    return {"calls": int(row[0]), "in": int(row[1]), "out": int(row[2]), "cost": float(row[3])}


def apply_footer_template(line: str, template: Optional[str]) -> str:
    """Wrap the footer line in ``template`` at ``{footer}`` (e.g. Slack ``"↳ _{footer}_"``). A template
    without ``{footer}`` is ignored rather than allowed to drop the data; plain ``str.replace`` so
    other braces in the template need no escaping."""
    if not line or not template or "{footer}" not in template:
        return line
    return template.replace("{footer}", line)


def format_runtime_footer(*, model: Optional[str], context_tokens: int,
                          context_length: Optional[int], cwd: Optional[str] = None,
                          turn_seconds: Optional[float] = None,
                          requested_model: Optional[str] = None, served_model: Optional[str] = None,
                          fields: Iterable[str] = _DEFAULT_FIELDS, ledger: Optional[dict] = None) -> str:
    """Render the footer line, or "" if no fields have data. Fields whose data is missing (and
    unknown field names) are skipped silently — a partial footer beats ``?%`` or empty slots."""
    def context_pct() -> str:
        if context_length and context_length > 0 and context_tokens >= 0:
            return f"{max(0, min(100, round((context_tokens / context_length) * 100)))}%"
        return ""

    def served() -> str:
        requested = requested_model or model
        alias = _model_short(requested)
        if served_model and served_model not in (alias, requested):
            return f"{alias} → {served_model}"
        return ""

    renderers = {
        "model": lambda: _model_short(model),
        "served_model": served,
        "context_pct": context_pct,
        # Skipped when the caller did not measure (None) or the value is negative.
        "latency": lambda: _format_latency(turn_seconds) if turn_seconds is not None and turn_seconds >= 0 else "",
        "cwd": lambda: _home_relative_cwd(cwd or _env_cwd()),
        "calls": lambda: f"{ledger['calls']} calls" if ledger else "",
        "tokens": lambda: f"{_human(ledger['in'])} in / {_human(ledger['out'])} out" if ledger else "",
        # ">=": a background review spawned after this reply is not in the ledger yet.
        "cost": lambda: f"≥${ledger['cost']:.2f}" if ledger else "",
    }
    return _SEP.join(v for field in fields if (render := renderers.get(field)) and (v := render()))


def build_footer_line(*, user_config: dict[str, Any] | None, platform_key: str | None,
                      model: Optional[str], context_tokens: int, context_length: Optional[int],
                      cwd: Optional[str] = None, turn_seconds: Optional[float] = None,
                      requested_model: Optional[str] = None, served_model: Optional[str] = None,
                      session_id: Optional[str] = None, db_path: Optional[str] = None) -> str:
    """Entry point for gateway/run.py: footer text, or "" when disabled / no data. Callers append it
    to the final response themselves, preserving a single blank line of separation.
    ``turn_seconds`` is the caller-measured (``time.monotonic()``) run duration; ``None`` skips the
    ``latency`` field. ``session_id`` + ``db_path`` feed the ledger fields (read only when one is set)."""
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    fields = cfg.get("fields") or _DEFAULT_FIELDS
    ledger = session_ledger(session_id, db_path) if any(f in _LEDGER_FIELDS for f in fields) else None
    line = format_runtime_footer(model=model, context_tokens=context_tokens,
                                 context_length=context_length, cwd=cwd, turn_seconds=turn_seconds,
                                 requested_model=requested_model, served_model=served_model,
                                 fields=fields, ledger=ledger)
    return apply_footer_template(line, cfg.get("template"))
