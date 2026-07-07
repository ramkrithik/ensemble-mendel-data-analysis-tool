"""Data Analysis Agent — an agentic AI assistant for CSV exploration.

The public surface is the :class:`~data_agent.agent.DataAnalysisAgent` class and the
:func:`~data_agent.config.load_config` helper. Everything else is an implementation
detail of the reason -> plan -> act -> observe -> respond loop.
"""

from data_agent.agent import DataAnalysisAgent
from data_agent.config import AppConfig, load_config

__all__ = ["DataAnalysisAgent", "AppConfig", "load_config"]
__version__ = "0.1.0"
