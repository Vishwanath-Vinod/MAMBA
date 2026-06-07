import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.parallel_scan import pscan

def inverse_softplus(x):
    """Numerically stable algebraic equivalent: log(e^x-1)=log(e^x(1-e^(-x)))=x+log(1-e^-x)"""
    return x + torch.log(-torch.expm1(-x))

def initialize_dt_projection(dt_rank,d_inner,dt_min=1e-3,dt_max=1e-1,dt_scale=1.0,dt_init_floor=1e-4,device=None,dtype=None,):
    """ Initialize the weights and biases for the projection network for dt which moves it from low dim dt_rank to d_inner unique for every channel."""
    layer = nn.Linear(dt_rank,d_inner,bias=True,device=device,dtype=dtype,)

    # Weight initialization
    std = dt_scale / math.sqrt(dt_rank)
    nn.init.uniform_(layer.weight,-std,std,)

    # Bias initialization: Sample dt values log-uniformly in [dt_min, dt_max] and initialize bias with inverse_softplus(dt)
    dt = torch.exp(torch.rand(d_inner, device=device, dtype=dtype)* (math.log(dt_max) - math.log(dt_min))+ math.log(dt_min)
                   ).clamp(min=dt_init_floor)
    with torch.no_grad():
        layer.bias.copy_(inverse_softplus(dt))
    layer.bias._no_reinit = True

    return layer

def selective_scan(x,dt,A,B,C,D=None,z=None,return_last_state=False,parallel=True):
    """
    A simple clean Selective scan with two implementations:
    1. Sequential recurrence : O(L)
    2. Parallel Blelloch scan : O(log L) depth
    """
    dtype_in = x.dtype
    x = x.float()
    dt = dt.float()
    batch_size, d_inner, L = x.shape
    d_state = A.shape[1]

    # (B,D,L,N) == (batch_size,d_inner,L,d_state)
    dA = torch.exp(torch.einsum("bdl,dn->bdln", dt, A))
    dBu = torch.einsum("bdl,bnl,bdl->bdln",dt,B,x)
    if parallel:
        H = pscan(dA.transpose(1, 2), dBu.transpose(1, 2),)  #Parallel scan expects (B, L, D, N)
        H = H.transpose(1, 2) # Back to (B, D, L, N)
        last_state = H[:, :, -1]
        y = torch.einsum("bdln,bnl->bdl", H, C)
    else:
        state = torch.zeros(batch_size,d_inner,d_state,device=x.device,dtype=x.dtype)
        outputs = []
        for t in range(L):
            state = (dA[:, :, t] * state+ dBu[:, :, t])
            yt = torch.einsum("bdn,bn->bd",state,C[:, :, t])
            outputs.append(yt)
        y = torch.stack(outputs, dim=-1)
        last_state = state

    if D is not None:
        y = y + x * D[:, None]
    if z is not None:
        y = y * F.silu(z)

    y = y.to(dtype_in)

    return (y, last_state) if return_last_state else y

class Mamba(nn.Module):
    def __init__(self,device=None,dtype=None,d_conv =4,d_model = 768, expand =2, d_state = 16,dt_rank = "auto",
                 dt_min=0.001,dt_max=0.1,dt_scale=1.0,dt_init_floor=1e-4,layer_idx = None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.act = nn.SiLU()
        self.d_model = d_model          # Input embedding dimension
        self.d_state = d_state          # State dimension of each SSM
        self.d_conv = d_conv            # Depthwise convolution kernel size
        self.expand = expand            # Expansion factor
        self.d_inner = int(expand * d_model) # Expanded hidden dimension used inside Mamba
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank # Low-rank dimension used for Δt parameterization
        self.layer_idx = layer_idx  # Index of the Mamba layer within the entire model

        #Input Projection: Expand and then split into 2 paths: Gating & S6
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False, **factory_kwargs)

        # Depthwise Convolution in S6 pathway: one filter for each channel with size d_conv
        self.conv1d = nn.Conv1d(in_channels=self.d_inner,out_channels=self.d_inner,bias=True,kernel_size=d_conv,groups=self.d_inner,padding=d_conv - 1,**factory_kwargs)
        
        #Extracting parameters: B (self.d_state),C (self.d_state) and Δt (self.dt_rank): different for each token
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        
        #Δt is diff for each channel for this project low size dt computed to full d_inner size using this dt_proj layer
        self.dt_proj = initialize_dt_projection(dt_rank=self.dt_rank,d_inner=self.d_inner,dt_min=dt_min,dt_max=dt_max,dt_scale=dt_scale,dt_init_floor=dt_init_floor,**factory_kwargs)
        
        #Output Projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)
        
        #S4D real initialization for A and D parameters: keep in fp32
        self.A_log = nn.Parameter(torch.log(torch.arange(1,self.d_state + 1,dtype=torch.float32,device=device,)).repeat(self.d_inner, 1)) 
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device,dtype=torch.float32))
        self.D._no_weight_decay = True

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states : (B, L, d_model)
        Modes:
            Training : inference_params = None: Entire sequence processed in parallel.
            Prefill  : inference_params.seqlen_offset == 0: Prompt is processed in parallel once to initialize conv_state and ssm_state.
            Decode   : inference_params.seqlen_offset > 0: Autoregressive generation using cached states from previous tokens.
        Inference Parameters:
            conv_state: (B, d_inner, d_conv): Stores the most recent d_conv inputs for every channel for the causal  convolution.
            ssm_state: (B, d_inner, d_state): Hidden state of SSM for every channel.
        """
        batch_size = hidden_states.shape[0]

        # Inference mode
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params,batch_size,)
            if inference_params.seqlen_offset > 0:
                output, _, _ = self.step(hidden_states,conv_state,ssm_state,)
                return output
            else:
                return self.process_sequence(hidden_states,conv_state=conv_state,ssm_state=ssm_state,)
        
        #Training mode
        return self.process_sequence(hidden_states)
    
    @property
    def A(self):
        """ A: (d_inner, d_state)"""
        return -torch.exp(self.A_log.float())

    def ssm_params(self, x, seqlen=None):
        """
        SSM Parameters: B,C and dt are token dependent.
            dt     : (B, d_inner, L)
            B      : (B, d_state, L)
            C      : (B, d_state, L)
        """
        x_proj = self.x_proj(x)
        dt, B, C = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)      # (B, dt_rank) -> (B, d_inner)
        dt = F.softplus(dt)       # ensure positivity

        if seqlen is None:
            return dt, B, C

        dt = rearrange(dt, "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) n -> b n l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) n -> b n l", l=seqlen).contiguous()
        return dt, B, C
        
    
    def process_sequence(self,hidden_states,conv_state=None,ssm_state=None,):
        """ Processes an entire sequence in parallel and intializes conv_state and ssm_state."""

        batch_size,seqlen,_ = hidden_states.shape
        
        xz = self.in_proj(hidden_states)      # (B,L,2*d_inner)
        x, z = xz.chunk(2, dim=-1)

        x = rearrange(x, "b l d -> b d l")    # (B,d_inner,L)
        z = rearrange(z, "b l d -> b d l")    # (B,d_inner,L)

        # Update convolution cache (prefill only), F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise. 
        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0))) 

        x = self.act(self.conv1d(x)[..., :seqlen]) 

        # SSM parameters
        dt, B, C = self.ssm_params(rearrange(x, "b d l -> (b l) d"), seqlen)
        dt_mean = dt.mean().item()
        dt_std  = dt.std().item()
        dt_min  = dt.min().item()
        dt_max  = dt.max().item()
        """
        if (dt_min < 1e-6 or dt_max > 1.0 or dt_mean < 5e-4 or dt_mean > 0.5 
            or torch.isnan(dt).any() or torch.isinf(dt).any()):
            print(f"[WARNING] dt statistics:"f" mean={dt_mean:.4e}"f" std={dt_std:.4e}"f" min={dt_min:.4e}"f" max={dt_max:.4e}")
        """
        # Selective scan
        y = selective_scan(x,dt,self.A,B,C,self.D.float(),z=z,return_last_state=(ssm_state is not None), parallel=True)

        # Update SSM cache (prefill only)
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state) 

        y = rearrange(y, "b d l -> b l d")

        return self.out_proj(y)
    
    def step(self, hidden_states, conv_state, ssm_state):
        "1 token at a time now: Autoregressive mode"

        assert hidden_states.shape[1] == 1
        dtype = hidden_states.dtype

        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B d_inner)

        #Convolution Step
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B d_inner d_conv)
        conv_state[:, :, -1] = x
        x_conv = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B d_inner)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x = self.act(x_conv).to(dtype=dtype)
        
        # SSM parameters
        dt,B,C = self.ssm_params(x)

        #Discretization: ZOH for A, approximate for B (Euler approxmn): easier to implement and good approximation
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, self.A))
        dB = torch.einsum("bd,bn->bdn", dt, B)

        #SSM step
        x = rearrange(x, "b d -> b d 1")
        ssm_state.copy_(ssm_state * dA +  x* dB)
        x = x.squeeze(-1)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)+ self.D.to(dtype) * x

        y = y * self.act(z)  # (B d_inner)
        out = self.out_proj(y) # (B d_model)
        return out.unsqueeze(1), conv_state, ssm_state
    
    def _get_states_from_cache(self,inference_params,batch_size,initialize_states=False,dtype=None):
        """
        Retrieve cached inference states for this layer. If no states exist yet, they are allocated and stored. Optionally resets existing states to zero.
        """
        assert self.layer_idx is not None

        cache = inference_params.key_value_memory_dict
        if self.layer_idx not in cache:
            conv_state, ssm_state = self.allocate_inference_cache(batch_size,dtype)
            cache[self.layer_idx] = conv_state, ssm_state

        conv_state, ssm_state = cache[self.layer_idx]

        if initialize_states:
            conv_state.zero_()
            ssm_state.zero_()

        return conv_state, ssm_state
    
    def allocate_inference_cache(self, batch_size, dtype=None, **kwargs):
        """
        Allocate and return inference caches for all Mamba layers. (If no states exist yet).Used only in generation (auto-regressive).
        """
        device = self.out_proj.weight.device
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device, dtype= dtype or self.conv1d.weight.dtype)
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=dtype or self.dt_proj.weight.dtype)
        return conv_state, ssm_state

