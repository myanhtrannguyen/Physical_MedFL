"""
Federated Learning core components.
"""

from .app_client import FlowerClient
from .app_server import server_fn

__all__ = [
    "FlowerClient",
    "server_fn"
]
