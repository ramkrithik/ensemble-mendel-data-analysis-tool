"""PC Build Agent — an agentic AI assistant that configures compatible PC builds.

Public surface: :class:`~pc_agent.agent.PCBuildAgent` (the reason -> plan -> act ->
observe -> respond loop) and :func:`~pc_agent.config.load_config`. Everything else
is an implementation detail.
"""

from pc_agent.agent import PCBuildAgent
from pc_agent.config import AppConfig, load_config

__all__ = ["PCBuildAgent", "AppConfig", "load_config"]
__version__ = "0.1.0"
