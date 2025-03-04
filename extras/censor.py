import os
import numpy as np
import torch

class Censor:
    def __init__(self):
        pass  # No safety checker model initialization

    def censor(self, images: list | np.ndarray) -> list | np.ndarray:
        return images  # Return images without any censorship

default_censor = Censor().censor
