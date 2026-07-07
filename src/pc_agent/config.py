"""Centralised, externalised configuration.

Every tunable — provider, model, temperature, loop limits, dataset location — is
read from the environment (optionally via a ``.env`` file) here and nowhere else.
No other module reaches for ``os.environ``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """Immutable snapshot of all runtime configuration."""

    provider: str  # "anthropic" | "bedrock"
    model: str
    temperature: float
    max_tokens: int
    agent_max_steps: int
    timeout_seconds: float
    max_retries: int
    trace_dir: Path
    data_dir: Path

    # Provider-specific
    anthropic_api_key: str | None
    aws_region: str

    def redacted(self) -> dict[str, object]:
        """Config as a dict safe for logging — the API key is never included."""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "agent_max_steps": self.agent_max_steps,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "data_dir": str(self.data_dir),
            "aws_region": self.aws_region if self.provider == "bedrock" else None,
        }


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Env var {name}={raw!r} is not a valid number") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Env var {name}={raw!r} is not a valid integer") from exc


def load_config(env_file: str | os.PathLike[str] | None = ".env") -> AppConfig:
    """Load configuration from the environment, layering in ``.env`` if present.

    Real environment variables win over ``.env`` (``override=False``).
    """
    if env_file is not None and Path(env_file).is_file():
        load_dotenv(env_file, override=False)

    provider = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()
    if provider not in ("anthropic", "bedrock"):
        raise ValueError(
            f"LLM_PROVIDER must be 'anthropic' or 'bedrock', got {provider!r}"
        )

    if provider == "bedrock":
        model = os.environ.get(
            "BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6"
        )
    else:
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    return AppConfig(
        provider=provider,
        model=model,
        temperature=_get_float("LLM_TEMPERATURE", 0.0),
        max_tokens=_get_int("LLM_MAX_TOKENS", 4096),
        agent_max_steps=_get_int("AGENT_MAX_STEPS", 12),
        timeout_seconds=_get_float("LLM_TIMEOUT_SECONDS", 60.0),
        max_retries=_get_int("LLM_MAX_RETRIES", 3),
        trace_dir=Path(os.environ.get("TRACE_DIR", "traces")),
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        aws_region=os.environ.get("AWS_REGION", "us-east-1"),
    )
