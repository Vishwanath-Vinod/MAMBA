import math
from dataclasses import dataclass,field
from typing import Union

@dataclass
class MambaConfig:
    d_model: int = 128
    n_layer: int = 4
    mlp_intermediate: int = 0
    vocab_size: int = 50277
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    dropout: float = 0.1

    pscan: bool = True # use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

    @property
    def mamba_kwargs(self):
        return {
            "d_model": self.d_model,
            "d_state": self.d_state,
            "expand": self.expand_factor,
            "d_conv": self.d_conv,
            "dt_rank": self.dt_rank,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "dt_scale": self.dt_scale,
            "dt_init_floor": self.dt_init_floor
        }
