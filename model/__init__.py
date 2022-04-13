# Deep Modular Graph Neural Network

from .model import Single, Hierachy
from .layers import GraphConvolutionLayer, SkipGraphConvolutionLayer

__all__ = ['Single', 'Hierachy',
           'GraphConvolutionLayer',
           'SkipGraphConvolutionLayer']
