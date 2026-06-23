"""Typed config loading: config.yaml for parameters, .env for secrets.

Import `settings` for the singleton, or call `load_settings()` to build a
fresh instance (e.g. in tests, pointed at a different config path).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class PathsConfig(BaseModel):
    raw_dir: str
    parquet_dir: str
    cache_dir: str
    results_file: str


class CalendarConfig(BaseModel):
    name: str = "NYSE"


class UniverseConfig(BaseModel):
    start_date: str
    end_date: str
    top_n_by_dollar_volume: int
    rebalance_freq: str
    min_price: float
    lookback_days_dollar_volume: int


class PublicationLagConfig(BaseModel):
    shares_outstanding_days: int
    price_days: int


class CostsConfig(BaseModel):
    commission_bps: float
    half_spread_bps: float
    market_impact_coefficient: float
    participation_rate_cap: float


class MomentumFactorConfig(BaseModel):
    lookback_months: int
    skip_months: int


class ReversalFactorConfig(BaseModel):
    lookback_months: int


class LowVolFactorConfig(BaseModel):
    lookback_days: int


class SizeFactorConfig(BaseModel):
    field: str


class FactorsConfig(BaseModel):
    momentum: MomentumFactorConfig
    reversal: ReversalFactorConfig
    low_vol: LowVolFactorConfig
    size: SizeFactorConfig


class ValidationConfig(BaseModel):
    min_pct_universe_months_with_delisted: float


class GlassboxConfig(BaseModel):
    seed: int
    paths: PathsConfig
    calendar: CalendarConfig
    universe: UniverseConfig
    publication_lag: PublicationLagConfig
    costs: CostsConfig
    factors: FactorsConfig
    validation: ValidationConfig

    @property
    def raw_dir(self) -> Path:
        return REPO_ROOT / self.paths.raw_dir

    @property
    def parquet_dir(self) -> Path:
        return REPO_ROOT / self.paths.parquet_dir

    @property
    def cache_dir(self) -> Path:
        return REPO_ROOT / self.paths.cache_dir

    @property
    def results_file(self) -> Path:
        return REPO_ROOT / self.paths.results_file


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    tiingo_api_key: str = ""
    fmp_api_key: str = ""


def load_settings(config_path: Path | str = DEFAULT_CONFIG_PATH) -> GlassboxConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return GlassboxConfig.model_validate(raw)


def load_secrets() -> Secrets:
    return Secrets()


settings = load_settings()
secrets = load_secrets()
