"""Gateway runtime-metadata footer.

Renders a compact footer showing runtime state (model, context %, cwd) and
appends it to the FINAL message of an agent turn when enabled.  Off by default
to keep replies minimal.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true                       # off by default
        fields: [model, context_pct, cwd]   # order shown; drop any to hide

Per-platform overrides live under ``display.platforms.<platform>.runtime_footer``.
Users can toggle the global setting with ``/footer on|off`` from both the CLI
and any gateway platform.

The footer is appended to the final response text in ``gateway/run.py`` right
before returning the response to the adapter send path — so it only lands on
the final message a user sees, not on tool-progress updates or streaming
partials.  When streaming is on and the final text has already been delivered
piecemeal, the footer is sent as a separate trailing message via
``send_trailing_footer()``.
"""

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
    """Drop ``vendor/`` prefix for readability (``openai/gpt-5.4`` → ``gpt-5.4``)."""
    if not model:
        return ""
    return model.rsplit("/", 1)[-1]


def resolve_footer_config(
    user_config: dict[str, Any] | None,
    platform_key: str | None = None,
) -> dict[str, Any]:
    """Resolve effective runtime-footer config for *platform_key*.

    Merge order (later wins):
        1. Built-in defaults (enabled=False)
        2. ``display.runtime_footer``
        3. ``display.platforms.<platform_key>.runtime_footer``
    """
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS), "template": ""}
    cfg = (user_config or {}).get("display") or {}

    global_cfg = cfg.get("runtime_footer")
    if isinstance(global_cfg, dict):
        if "enabled" in global_cfg:
            resolved["enabled"] = bool(global_cfg.get("enabled"))
        if isinstance(global_cfg.get("fields"), list) and global_cfg["fields"]:
            resolved["fields"] = [str(f) for f in global_cfg["fields"]]
        if isinstance(global_cfg.get("template"), str):
            resolved["template"] = global_cfg["template"]

    if platform_key:
        platforms = cfg.get("platforms") or {}
        plat_cfg = platforms.get(platform_key)
        if isinstance(plat_cfg, dict):
            plat_footer = plat_cfg.get("runtime_footer")
            if isinstance(plat_footer, dict):
                if "enabled" in plat_footer:
                    resolved["enabled"] = bool(plat_footer.get("enabled"))
                if isinstance(plat_footer.get("fields"), list) and plat_footer["fields"]:
                    resolved["fields"] = [str(f) for f in plat_footer["fields"]]
                if isinstance(plat_footer.get("template"), str):
                    resolved["template"] = plat_footer["template"]

    return resolved


_LEDGER_FIELDS = frozenset({"calls", "tokens", "cost"})


def _human(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def session_ledger(session_id: Optional[str], db_path: Optional[str]) -> Optional[dict]:
    """Whole-session usage totals from ``session_model_usage``, or None.

    Reads the LEDGER, not the agent's in-memory counters. ``session_*`` on the
    agent start at 0.0 when the agent object is created and are never restored
    from the DB, so after a gateway restart or a cache eviction they mean "since
    this process picked the session up" — a window mislabelled as a session
    total, which is precisely the class of wrong footer this feature exists to
    replace. The ledger is keyed by session and spans every task (main loop,
    approval, compression, title generation, background review), and the agent
    flushes its queued token deltas at turn finalize, before the gateway builds
    this footer.

    It is still a FLOOR for the current turn: a background review is spawned
    after the final response, so its cost lands in the next turn's footer.

    Read-only URI connection — the gateway owns the writer; WAL permits readers.
    Any failure returns None and the ledger fields are skipped silently, same as
    every other field whose data is missing.
    """
    if not session_id or not db_path:
        return None
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
        try:
            row = con.execute(
                "SELECT COALESCE(SUM(api_call_count),0),"
                " COALESCE(SUM(input_tokens + cache_read_tokens + cache_write_tokens),0),"
                " COALESCE(SUM(output_tokens),0),"
                " COALESCE(SUM(estimated_cost_usd),0)"
                " FROM session_model_usage WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        finally:
            con.close()
    except Exception:
        return None
    if not row or not row[0]:
        return None
    return {"calls": int(row[0]), "in": int(row[1]), "out": int(row[2]), "cost": float(row[3])}


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    fields: Iterable[str] = _DEFAULT_FIELDS,
    ledger: Optional[dict] = None,
) -> str:
    """Render the footer line, or return "" if no fields have data.

    Fields are skipped silently when their underlying data is missing — a
    partially-populated footer is better than a line with ``?%`` or empty slots.
    """
    parts: list[str] = []
    for field in fields:
        if field == "model":
            m = _model_short(model)
            if m:
                parts.append(m)
        elif field == "context_pct":
            if context_length and context_length > 0 and context_tokens >= 0:
                pct = max(0, min(100, round((context_tokens / context_length) * 100)))
                parts.append(f"{pct}%")
        elif field == "cwd":
            rel = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
            if rel:
                parts.append(rel)
        elif field == "calls" and ledger:
            parts.append(f"{ledger['calls']} calls")
        elif field == "tokens" and ledger:
            parts.append(f"{_human(ledger['in'])} in / {_human(ledger['out'])} out")
        elif field == "cost" and ledger:
            # ">=" because a background review spawned after this reply is not
            # yet in the ledger; it shows up in the next turn's footer.
            parts.append(f"≥${ledger['cost']:.2f}")
        # Unknown field names are silently ignored.

    if not parts:
        return ""
    return _SEP.join(parts)


def build_footer_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    session_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> str:
    """Top-level entry point used by gateway/run.py.

    Returns the footer text (empty string when disabled or no data).  Callers
    append this to the final response themselves, preserving a single blank
    line of separation.
    """
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    fields = cfg.get("fields") or _DEFAULT_FIELDS
    ledger = (session_ledger(session_id, db_path)
              if any(f in _LEDGER_FIELDS for f in fields) else None)
    line = format_runtime_footer(
        model=model,
        context_tokens=context_tokens,
        context_length=context_length,
        cwd=cwd,
        fields=fields,
        ledger=ledger,
    )
    return apply_footer_template(line, cfg.get("template"))


def apply_footer_template(line: str, template: Optional[str]) -> str:
    """Wrap the footer line in ``template``, where ``{footer}`` marks the line.

    Lets a platform style the footer in its own markup, e.g. Slack
    ``"↳ _{footer}_"`` for a marker plus italics. A template without
    ``{footer}`` is ignored rather than allowed to drop the data; plain
    ``str.replace`` so other braces in the template need no escaping.
    """
    if not line or not template or "{footer}" not in template:
        return line
    return template.replace("{footer}", line)
