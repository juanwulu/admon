"""Deep Modular Graph Neural Network"""

from .layers import GCN
from .model import Single, Hierachy, DMoN

__all__ = ['Single', 'Hierachy', 'DMoN', 'GCN']
