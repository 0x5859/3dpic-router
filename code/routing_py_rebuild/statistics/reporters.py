"""Serialize a recorder.finalize() dict to run_report.{json,md}.

The serialization layer never reads :class:`RunRecorder` state directly; it
only consumes the dict, which keeps it swappable (e.g. SQLite-backed sinks
later) without touching this file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import jsonschema

# code/routing_py_rebuild/statistics/reporters.py
#   parents[0] = statistics/
#   parents[1] = routing_py_rebuild/
#   parents[2] = code/
_REPO_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schema"
_REPORT_SCHEMA = _REPO_SCHEMA_DIR / "run_report.schema.json"


def _validate(report: dict[str, Any]) -> None:
    """Validate ``report`` against ``run_report.schema.json`` v1.0.

    Raises ``jsonschema.ValidationError`` on mismatch.
    """
    with open(_REPORT_SCHEMA) as f:
        schema = json.load(f)
    jsonschema.validate(instance=report, schema=schema)


def write_run_report(
    report: dict[str, Any],
    out_dir: str | os.PathLike,
) -> tuple[str, str]:
    """Write ``run_report.{json,md}`` into ``out_dir``.

    Returns ``(json_path, md_path)``. Validates ``report`` against the
    schema **before** writing — caller sees a ValidationError instead of
    a half-written file pair.
    """
    _validate(report)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "run_report.json"
    md_path = out / "run_report.md"

    with open(json_path, "w") as f:
        # allow_nan=False ⇒ raise on NaN/Infinity rather than emit non-strict
        # JSON tokens that downstream parsers (esp. C++ nlohmann::json at
        # M5 parity) reject. RunRecorder.on_iter already filters non-finite
        # losses upstream, so this is a defense-in-depth check.
        json.dump(report, f, indent=2, allow_nan=False)
    md_path.write_text(_render_md(report))

    return str(json_path), str(md_path)


def _render_md(report: dict[str, Any]) -> str:
    """Render the human-readable companion. Mirrors §1-1-a 输出格式."""
    summary = report["summary"]
    timings = report["timings"]
    config = report["config"]

    lines: list[str] = [
        f"# Run report — {report['run_id']}",
        "",
        f"`schema_version`: {report['schema_version']}",
        "",
        "## Summary",
        "",
        f"- iterations: **{summary['iterations']}**",
        f"- initial loss: **{summary['initial_loss']:.6g}**",
        f"- final loss (best): **{summary['final_loss']:.6g}**",
        f"- relative drop: **{summary['relative_drop'] * 100:.2f}%**",
        f"- best at iter: **{summary['best_at_iter']}**",
        "",
        "## Timings (ms)",
        "",
    ]
    if timings:
        for key in sorted(timings):
            lines.append(f"- `{key}`: {timings[key]}")
    else:
        lines.append("_(no phases recorded)_")

    lines.extend(
        [
            "",
            "## Config",
            "",
            "```json",
            json.dumps(config, indent=2, sort_keys=True),
            "```",
            "",
            "_See `run_report.json` for the machine-readable trace._",
            "",
        ]
    )
    return "\n".join(lines)
