"""M0 acceptance tests: config loads, contracts documented, results schema works."""

from __future__ import annotations

import pytest

import glassbox
from glassbox.engine import asof
from glassbox.results import ResultsStore
from glassbox.settings import load_settings

CORE_INVARIANT = (
    "No computation may read any datum whose knowable-date is later than "
    "the simulation's current as-of clock."
)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def test_core_invariant_documented_in_package_docstring():
    assert _normalize(CORE_INVARIANT) in _normalize(glassbox.__doc__)


def test_core_invariant_documented_in_asof_contract():
    assert _normalize(CORE_INVARIANT) in _normalize(asof.__doc__)


def test_config_loads():
    cfg = load_settings()
    assert cfg.seed == 42
    assert cfg.universe.top_n_by_dollar_volume > 0
    assert cfg.universe.start_date < cfg.universe.end_date


def test_config_paths_resolve_under_repo_root():
    cfg = load_settings()
    assert cfg.parquet_dir.name == "parquet"
    assert cfg.results_file.name == "results.json"


def test_results_store_set_target_then_result(tmp_path):
    rs = ResultsStore()
    rs.set_target("dummy.metric", 1.5)
    rs.set_result("dummy.metric", 1.7)
    assert rs.metrics["dummy.metric"].target == 1.5
    assert rs.metrics["dummy.metric"].result == 1.7


def test_results_store_refuses_result_without_target():
    rs = ResultsStore()
    with pytest.raises(KeyError):
        rs.set_result("no_target_set", 1.0)


def test_results_store_set_result_never_overwrites_target(tmp_path):
    rs = ResultsStore()
    rs.set_target("dummy.metric", 1.5)
    rs.set_result("dummy.metric", 1.7)
    rs.set_result("dummy.metric", 1.9)
    assert rs.metrics["dummy.metric"].target == 1.5


def test_results_store_roundtrip(tmp_path):
    path = tmp_path / "results.json"
    rs = ResultsStore()
    rs.set_target("a", 1.0)
    rs.set_result("a", 2.0)
    rs.save(path)
    loaded = ResultsStore.load(path)
    assert loaded.metrics["a"].target == 1.0
    assert loaded.metrics["a"].result == 2.0


def test_data_provider_protocol_is_runtime_checkable():
    from glassbox.data.provider import DataProvider

    assert hasattr(DataProvider, "get_price_history")
    assert hasattr(DataProvider, "get_universe_symbols")
    assert hasattr(DataProvider, "is_point_in_time")
