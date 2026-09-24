"""The default loss model is one set of numbers everywhere.

Intralayer crossing 0.1 dB, taper 0.05 dB, interlayer crossing 0.001 dB
per event: ``core.DEFAULT_LOSS_*`` feeds the graph class, ``api``, and the
CLI; the JSON reader falls back to the same crossing loss; and the C++
harness (``graph.hpp`` / ``main.cpp`` env defaults) uses the same values.
"""
from __future__ import annotations

import inspect
import json
import subprocess

import pytest
from routing_py_rebuild import api, core
from routing_py_rebuild.__main__ import _build_parser
from routing_py_rebuild.plotting.plot_data import PlotData

from tests.parity_harness import CPP_BINARY, hermetic_env

EXPECTED = {"loss_crossing": 0.1, "loss_taper": 0.05, "loss_interlayercrossing": 0.001}


def test_core_constants_are_the_default_loss_model():
    assert {
        "loss_crossing": core.DEFAULT_LOSS_CROSSING,
        "loss_taper": core.DEFAULT_LOSS_TAPER,
        "loss_interlayercrossing": core.DEFAULT_LOSS_INTERLAYERCROSSING,
    } == EXPECTED


@pytest.mark.parametrize("func", [
    core.SiNInterconnectionGraph.__init__, api.make_graph, api.run_optimization,
], ids=["SiNInterconnectionGraph", "make_graph", "run_optimization"])
def test_python_entry_points_share_the_defaults(func):
    params = inspect.signature(func).parameters
    assert {name: params[name].default for name in EXPECTED} == EXPECTED


def test_cli_shares_the_defaults():
    args = _build_parser().parse_args(["optimize"])
    assert {name: getattr(args, name) for name in EXPECTED} == EXPECTED


def test_default_run_records_the_loss_model(tmp_path):
    res = api.run_optimization(k=8, maxiter=2, output_dir=str(tmp_path), plot=False,
                               run_loss_analysis=False)
    assert "cl_0.10_tl_0.05_itl_0.001_nodes_8" in res["json_path"]
    with open(res["json_path"]) as f:
        saved = json.load(f)
    gp = saved["General Parameters"]
    assert (gp["Loss of Crossing"], gp["Loss of Taper"],
            gp["Loss of Interlayer Crossing"]) == (0.1, 0.05, 0.001)

    # A file without the crossing loss (it is optional in the schema)
    # reads back with the same default.
    del gp["Loss of Crossing"]
    stripped = tmp_path / "stripped.json"
    stripped.write_text(json.dumps(saved))
    assert PlotData.from_json(str(stripped)).loss_crossing == core.DEFAULT_LOSS_CROSSING


def test_cpp_harness_shares_the_defaults(tmp_path):
    if not CPP_BINARY.exists():
        pytest.skip(f"C++ binary not built at {CPP_BINARY}.")
    env = hermetic_env()  # CL_VALUES / TL_VALUES / ITL_VALUES unset -> C++ defaults
    env.update({"NODES_MIN": "1", "NODES_MAX": "1", "DUALSA_ITER": "1",
                "OUTPUT_DIR": str(tmp_path) + "/"})
    subprocess.run([str(CPP_BINARY)], env=env, check=True, capture_output=True, timeout=300)
    out = tmp_path / "cl_0.10_tl_0.05_itl_0.001_nodes_4" / "subgraphsdata.json"
    gp = json.loads(out.read_text())["General Parameters"]
    assert (gp["Loss of Crossing"], gp["Loss of Taper"],
            gp["Loss of Interlayer Crossing"]) == (0.1, 0.05, 0.001)
