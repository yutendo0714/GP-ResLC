"""Scratch GP-ResLC modules for jointly trained semantic/residual decomposition."""

from .vq_autoencoder import ScratchVQAutoencoder, VectorQuantizer
from .residual_autoencoder import ScratchResidualBottleneck, ScratchProgressiveResidualBottleneck
from .glc_latent_residual import GLCLatentResidualBottleneck
from .discriminator import PatchDiscriminator

__all__ = ["ScratchVQAutoencoder", "VectorQuantizer", "ScratchResidualBottleneck",
    "ScratchProgressiveResidualBottleneck", "GLCLatentResidualBottleneck", "PatchDiscriminator"]
