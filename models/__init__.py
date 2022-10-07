from .AE import AutoEncoder
from .VAE import VariationalAutoEncoder
from .GAN import GAN
from .WGAN import WGAN
from .WGANGP import WGANGP
from .CycleGAN import CycleGAN
from .MuseGAN import MuseGAN
from .RNNAttention import *

__all__ = ["AutoEncoder", "VariationalAutoEncoder", "GAN", "WGAN",
           "WGANGP", "CycleGAN", "MuseGAN"]

__all__.extend(RNNAttention.__all__)
