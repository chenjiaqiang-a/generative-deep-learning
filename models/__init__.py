from .AE import AutoEncoder
from .VAE import VariationalAutoEncoder
from .GAN import GAN
from .WGAN import WGAN
from .WGANGP import WGANGP

__all__ = ["AutoEncoder", "VariationalAutoEncoder", "GAN", "WGAN",
           "WGANGP"]