"""GP-ResLC research modules layered on top of the official GLC code."""

from .perceptual_gate import PerceptualGate
from .prior_predictor import PriorPredictor, train_forward

__all__ = ["PerceptualGate", "PriorPredictor", "train_forward"]
