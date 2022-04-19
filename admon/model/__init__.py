"""Deep Modular Graph Neural Network"""

from .layers import DMoN, GCNLayer
from .model import Single, Hierachy

__all__ = ['Single', 'Hierachy', 'DMoN', 'GCNLayer']
