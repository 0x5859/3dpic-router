"""Plotting-only data container.

:class:`PlotData` is the single object every plotting function consumes.
It bundles exactly the information needed to render a figure — positions,
per-layer subgraphs, the complete-graph baseline, loss-analysis stats,
and ``loss_crossing`` (the only loss parameter plot code reads, for the
overlay-row histogram annotation) — and is constructible either from a
saved JSON (:meth:`from_json`) or from a live graph (:meth:`from_graph`,
used as an escape hatch for tests; strictly read-only).

Plotting code never imports :mod:`core` and never mutates a
:class:`SiNInterconnectionGraph`; the JSON file is the contract.

M3 (REFACTOR_GOALS.md §3-1 兼容性策略): :meth:`from_json` reads
``schema_version`` and applies the loader policy from §3-1:

- ``schema_version`` missing or starting with ``"1."`` → legacy path:
  soft-default ``L=2``, ``edge_coupler_layer=0``, ``perimeter_layer=0``,
  ``waveguides_per_link=1``, ``crosstalk=None``; emit the
  ``waveguides_per_link=1`` warning since the loss model still
  accumulates the implicit 2× and the stats convention is now
  ``"geometric"`` (§4 M3 附注 + §7 Q-e).
- ``schema_version`` starting with ``"2."`` → strict path: each new
  field (``L`` / ``edge_coupler_layer`` / ``perimeter_layer`` /
  ``layer_pitch_um`` / ``waveguides_per_link``) is required; missing →
  ``ValueError``. ``waveguides_per_link`` is range-checked per §6 T9.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
import networkx as nx
import numpy as np


# REFACTOR_GOALS.md §3-3 read-time validation. ``functools.lru_cache`` makes
# the first-load-wins memoization atomic at the function-call boundary so
# parallel pytest-xdist workers don't double-read the file
# (opus-review-3 P2-D). Path is resolved from this file (plotting/) up two
# parents to ``code/`` and then into ``code/schema/``.
from functools import lru_cache as _lru_cache

_SUBGRAPHS_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schema" / "subgraphsdata.schema.json"
)


@_lru_cache(maxsize=1)
def _load_subgraphs_schema() -> dict:
    with open(_SUBGRAPHS_SCHEMA_PATH) as f:
        return json.load(f)


# REFACTOR_GOALS.md §4 M3 附注 "Warning 策略" (post-2026-05-15 revision)
# — kept in sync with the constants in ``core.py`` (text duplicated
# rather than imported to keep the §1-3-c "plotting must not import
# core" boundary intact). wpl=1 is now the documented v2.0 default; the
# notice is one-time per process to avoid stderr spam.
_WPL_ONE_INFO = (
    "waveguides_per_link=1 (default): stats use the geometric crossing "
    "convention (matches paper reporting). The per-edge loss formula keeps "
    "an implicit 2× from endpoint double-counting; pass "
    "waveguides_per_link=2 for the physical convention aligned with the "
    "loss model. See REFACTOR_GOALS.md §7 Q-e."
)
_WPL_EXPERIMENTAL_WARNING = (
    "waveguides_per_link not in {{1, 2}} is experimental: stats/loss "
    "semantics may diverge until §7 Q-e is decided. wpl={value}."
)

# Mirror core._WPL_PHYSICAL_MAX; kept verbatim per the §1-3-c
# "plotting must not import core" boundary (opus-review-3 P2-E).
_WPL_PHYSICAL_MAX = 32

# Loader-path one-time emit flag (mirror of the core.py flag). Kept
# separate because §1-3-c forbids plotting from importing core; the
# test conftest resets both flags before each test so isolation holds.
_wpl_one_notice_emitted = False


def _emit_wpl_one_notice_once() -> None:
    """Loader-path mirror of ``core._emit_wpl_one_notice_once``."""
    global _wpl_one_notice_emitted
    if _wpl_one_notice_emitted:
        return
    _wpl_one_notice_emitted = True
    warnings.warn(_WPL_ONE_INFO, stacklevel=4)


def _reset_wpl_notice_for_tests() -> None:
    """Reset the one-time flag. Test-only — invoked by ``conftest.py``."""
    global _wpl_one_notice_emitted
    _wpl_one_notice_emitted = False


def _validate_loaded_wpl(wpl: Any, *, source: str) -> int:
    """Loader-side validation for ``waveguides_per_link``.

    Mirrors the policy in REFACTOR_GOALS.md §4 M3 附注 "Warning 策略"
    (post-2026-05-15) and §6 T9: positive integer required (booleans
    rejected even though ``isinstance(True, int)`` is True in Python);
    silent for 2; info-once for 1; warning for {3, 4, ..., 32};
    ``ValueError`` for 0 / negative / non-int / > ``_WPL_PHYSICAL_MAX``.

    ``source`` is interpolated into the error message ("legacy soft-default"
    vs. "v2.0 'General Parameters'") so the failure mode is obvious.
    """
    # Match core._validate_waveguides_per_link policy: bool rejected even
    # though ``isinstance(True, int)`` is True; accept ``np.integer`` so a
    # caller plumbing wpl through ``PlotData.from_graph(graph)`` with
    # numpy-typed integers does not spuriously fail at the loader edge.
    if isinstance(wpl, bool) or not isinstance(wpl, (int, np.integer)):
        raise ValueError(
            f"{source}: waveguides_per_link must be a positive integer, "
            f"got {wpl!r} (type {type(wpl).__name__})."
        )
    if wpl <= 0:
        raise ValueError(
            f"{source}: waveguides_per_link must be >= 1; got {wpl}."
        )
    if wpl > _WPL_PHYSICAL_MAX:
        raise ValueError(
            f"{source}: waveguides_per_link={wpl} exceeds physical bound "
            f"{_WPL_PHYSICAL_MAX}; typical photonic interconnects use 1–8."
        )
    if wpl == 1:
        _emit_wpl_one_notice_once()
    elif wpl != 2:
        warnings.warn(
            _WPL_EXPERIMENTAL_WARNING.format(value=wpl), stacklevel=3
        )
    return int(wpl)


@dataclass
class PlotData:
    """Self-contained snapshot consumed by every plotting function."""

    k: int
    positions: dict[int, tuple[float, float]]
    layers: list[nx.Graph]
    complete_graph: nx.Graph
    loss_analysis: dict[str, Any]
    loss_crossing: float
    general_params: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None
    # Whether ``source_path`` denotes a directory (True; from
    # :meth:`from_graph`, where it is the per-config output dir) or a
    # file (False; from :meth:`from_json`, the source JSON). Recorded by
    # the constructor instead of probed at read time so the resolution
    # is independent of whether the directory exists on disk yet — see
    # the note on :attr:`filepath`.
    source_is_dir: bool = False

    @property
    def filepath(self) -> str | None:
        """Default output directory derived from ``source_path``.

        Two cases, distinguished by the constructor-recorded
        :attr:`source_is_dir` flag (NOT by probing the filesystem — the
        per-config output directory may not exist yet when plotting from
        an in-memory graph without a JSON write):

          - ``source_path`` is a file (e.g. the JSON loaded by
            :meth:`from_json`, ``source_is_dir=False``) → returns
            ``os.path.dirname(os.path.abspath(source_path))``
          - ``source_path`` is a directory (e.g. ``graph.filepath``
            passed into :meth:`from_graph`, ``source_is_dir=True``) →
            returns it unchanged

        Returns ``None`` when no source path is known.
        """
        if self.source_path is None:
            return None
        if self.source_is_dir:
            return self.source_path
        return os.path.dirname(os.path.abspath(self.source_path))

    @classmethod
    def from_json(cls, json_path: str) -> "PlotData":
        """Load the self-contained schema written by
        :meth:`core.SiNInterconnectionGraph.save_subgraphs_to_json`.

        Applies the schema_version policy from REFACTOR_GOALS.md §3-1.

        Raises
        ------
        ValueError
            If ``positions`` or ``complete_graph`` are missing — these are
            written by the current schema; older files predate the
            plotting decoupling and must be regenerated.
            For ``schema_version`` starting with ``"2."``, additional
            ``ValueError`` is raised when ``L`` / ``edge_coupler_layer`` /
            ``perimeter_layer`` / ``layer_pitch_um`` / ``waveguides_per_link``
            are missing or out of range (M3 / §3-1 兼容性策略).
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        # REFACTOR_GOALS.md §3-3: read-time validation. Catches malformed
        # v1.x soft-default cases AND missing v2.0 required fields. The
        # schema's allOf-if/then dispatches on schema_version; legacy v1.x
        # without the new fields is accepted, v2.0 missing required fields
        # raises a ``jsonschema.ValidationError`` here (T5 / T9).
        jsonschema.validate(instance=data, schema=_load_subgraphs_schema())

        schema_version = str(data.get("schema_version", "1.0"))
        params = dict(data.get("General Parameters", {}))

        # M3 schema-version dispatch — see REFACTOR_GOALS.md §3-1 兼容性策略.
        is_v2 = schema_version.startswith("2.")
        if is_v2:
            required = (
                "L",
                "edge_coupler_layer",
                "perimeter_layer",
                "layer_pitch_um",
                "waveguides_per_link",
            )
            missing = [k for k in required if k not in params]
            if missing:
                raise ValueError(
                    f"{json_path}: schema_version={schema_version!r} requires "
                    f"'General Parameters' fields {missing}; got "
                    f"{sorted(params.keys())}."
                )
            wpl = _validate_loaded_wpl(
                params["waveguides_per_link"],
                source=f"{json_path} (v2.0 General Parameters)",
            )
            params["waveguides_per_link"] = wpl
        else:
            # Legacy v1.x soft-defaults per §3-1 兼容性策略 first bullet.
            params.setdefault("L", 2)
            params.setdefault("edge_coupler_layer", 0)
            params.setdefault("perimeter_layer", 0)
            params.setdefault("layer_pitch_um", None)
            soft_default_wpl = "waveguides_per_link" not in params
            if soft_default_wpl:
                params["waveguides_per_link"] = 1
            # Still validate the value — a malformed legacy file with
            # ``wpl=0`` or ``"two"`` should fail loudly.
            wpl = _validate_loaded_wpl(
                params["waveguides_per_link"],
                source=(
                    f"{json_path} (v1.x soft-default)"
                    if soft_default_wpl
                    else f"{json_path} (v1.x explicit)"
                ),
            )
            params["waveguides_per_link"] = wpl

        # opus-review-3 P2-A: pass the top-level ``crosstalk`` blob through
        # to ``general_params["crosstalk"]`` so M4 readers (and downstream
        # plotting code) can see the rank-3 tensor without re-loading the
        # JSON file. M3 writes never emit ``crosstalk``; the field is
        # only present on M4-written files. Default to None so existing
        # M3 fixtures behave unchanged.
        params["crosstalk"] = data.get("crosstalk")

        raw_positions = data.get("positions")
        if not raw_positions:
            raise ValueError(
                f"{json_path}: 'positions' key missing. This JSON predates "
                "the plotting decoupling — re-run the optimizer to regenerate."
            )
        positions = {
            int(idx): (float(coord[0]), float(coord[1]))
            for idx, coord in raw_positions.items()
        }
        k = int(params.get("k", len(positions)))

        # Range-check multi-layer fields once we know L (v2.0 path; legacy
        # soft-defaults are already valid).
        L = int(params["L"])
        if L < 1:
            raise ValueError(f"{json_path}: L must be >= 1; got {L}.")
        for fld in ("edge_coupler_layer", "perimeter_layer"):
            v = int(params[fld])
            if not (0 <= v < L):
                raise ValueError(
                    f"{json_path}: {fld}={v} out of range [0, L) with L={L}."
                )

        layers: list[nx.Graph] = []
        layer_keys = sorted(
            (key for key in data.keys() if key.startswith("Layer_")),
            key=lambda s: int(s.split("_")[1]),
        )
        for layer_key in layer_keys:
            sg = nx.Graph()
            sg.add_nodes_from(range(k))
            for edge_triplet in data[layer_key]["edges"]:
                u, v, attr = edge_triplet
                sg.add_edge(int(u), int(v), **dict(attr))
            layers.append(sg)

        complete_edges = data.get("complete_graph")
        if not complete_edges:
            raise ValueError(
                f"{json_path}: 'complete_graph' key missing. This JSON predates "
                "the plotting decoupling — re-run the optimizer to regenerate."
            )
        complete = nx.Graph()
        complete.add_nodes_from(range(k))
        for edge_triplet in complete_edges:
            u, v, attr = edge_triplet
            complete.add_edge(int(u), int(v), **dict(attr))

        loss_analysis = data.get("loss_analysis", {})
        loss_crossing = float(params.get("Loss of Crossing", 0.3))

        return cls(
            k=k,
            positions=positions,
            layers=layers,
            complete_graph=complete,
            loss_analysis=loss_analysis,
            loss_crossing=loss_crossing,
            general_params=params,
            source_path=json_path,
            # The JSON file itself; ``filepath`` returns its directory.
            source_is_dir=False,
        )

    @classmethod
    def from_graph(cls, graph) -> "PlotData":
        """Snapshot a :class:`core.SiNInterconnectionGraph` **without mutating it**.

        Requires the graph to be already analyzed — i.e. ``sub_G`` populated
        and ``loss_analysis`` non-empty. The caller is responsible for
        calling ``graph.analyze_loss()`` first; plotting is read-only and
        must not trigger graph-side computations.

        Provided as an escape hatch for tests and ad-hoc in-memory use.
        Production flow goes through JSON via :meth:`from_json`.

        Raises
        ------
        ValueError
            If the graph is not in analyzed state.
        """
        if not graph.sub_G or not graph.loss_analysis:
            raise ValueError(
                "PlotData.from_graph requires an analyzed graph: call "
                "graph.analyze_loss() first (it populates sub_G and "
                "loss_analysis). Alternatively, save the graph to JSON and "
                "use PlotData.from_json(path)."
            )
        return cls(
            k=int(graph.k),
            positions={int(idx): tuple(coord) for idx, coord in graph.positions.items()},
            layers=[sg.copy() for sg in graph.sub_G],
            complete_graph=graph._G_Planar.copy(),
            loss_analysis=dict(graph.loss_analysis),
            loss_crossing=float(graph.loss_crossing),
            general_params={
                "k": int(graph.k),
                "L": int(getattr(graph, "L", 2)),
                "edge_coupler_layer": int(getattr(graph, "edge_coupler_layer", 0)),
                "perimeter_layer": int(getattr(graph, "perimeter_layer", 0)),
                "layer_pitch_um": float(getattr(graph, "layer_pitch_um", 1.2)),
                "waveguides_per_link": int(getattr(graph, "waveguides_per_link", 1)),
                "Loss of Taper": float(graph.loss_taper),
                "Loss of Crossing": float(graph.loss_crossing),
                "Loss of Interlayer Crossing": float(graph.loss_interlayercrossing),
                "Loss of Intralayer Crosstalk": graph.loss_intralayer_crosstalk,
                "Loss of Interlayer Crosstalk": graph.loss_interlayer_crosstalk,
                "coherence_model": getattr(graph, "coherence_model", "incoherent_v1"),
                "polarization": getattr(graph, "polarization", "TE0_only"),
                # M4 opus-review P2-2: ``from_json`` always exposes the
                # ``crosstalk`` key (default None per loader passthrough).
                # ``from_graph`` does the same so consumers can rely on
                # ``params["crosstalk"]`` (KeyError-free) regardless of
                # which constructor produced the snapshot. The
                # orchestrator injects the actual payload into this slot
                # post-construction when M4 has run; in the no-crosstalk
                # baseline the slot stays None.
                "crosstalk": None,
            },
            source_path=graph.filepath,
            # ``graph.filepath`` is the per-config output directory
            # (``<output_dir>/cl_..._nodes_<k>/``), not a file. Record
            # that explicitly so ``filepath`` resolves to this directory
            # even before it is created on disk (e.g. save_json=False).
            source_is_dir=True,
        )


__all__ = ["PlotData"]
