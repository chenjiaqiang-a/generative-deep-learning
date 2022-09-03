from .AE import AutoEncoder
from .VAE import VariationalAutoEncoder
from .GAN import GAN
from .WGAN import WGAN
from .WGANGP import WGANGP
from .CycleGAN import CycleGAN

__all__ = ["AutoEncoder", "VariationalAutoEncoder", "GAN", "WGAN",
           "WGANGP", "CycleGAN"]
