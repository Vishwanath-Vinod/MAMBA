import math
import torch
import torch.nn.functional as F

"""
This is the parallelized in-place Pytorch implementation of Blleloch scan obtained from  https://github.com/alxndrTL/mamba.py
and modified for compatibility with our MAMBA barebones implementation.
It modifies X in place by populating with values H[t] = A[t] * H[t-1] + X[t]. It supports only for L that is a power of 2.
"""

def npo2(len):
    """
    Returns the next power of 2 above len
    """
    return 2 ** math.ceil(math.log2(len))

def pad(X):
    """
    Pads input length dim to the next power of 2. (B, L, D, N) -> (B, npo2(L), D, N)
    """
    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)

class ParallelScan(torch.autograd.Function):
    @staticmethod
    def compose(A_left, X_left, A_right, X_right):
        """
        Composition used in scan for MAMBA recurrence.
        Ar <-- Ar*Al    Xr <-- Ar*Xl + Xr
        """
        X_right.add_(A_right * X_left)
        A_right.mul_(A_left)

    @staticmethod
    def scan(A, X):
        """
        Performs the in-place Blleloch scan. A : (B, D, L, N) X : (B, D, L, N)
        """
        B, D, L, _ = A.shape
        num_steps = int(math.log2(L))

        # UPSWEEP (last 2 levels done explicitly)
        A_level = A
        X_level = X
        for _ in range(num_steps - 2):
            length = A_level.size(2)
            A_level = A_level.view(B, D, length // 2, 2, -1)
            X_level = X_level.view(B, D, length // 2, 2, -1)
            ParallelScan.compose(A_level[:, :, :, 0],X_level[:, :, :, 0],A_level[:, :, :, 1],X_level[:, :, :, 1],)
            A_level = A_level[:, :, :, 1]
            X_level = X_level[:, :, :, 1]

        if X_level.size(2) == 4:
            ParallelScan.compose(A_level[:, :, 0],X_level[:, :, 0],A_level[:, :, 1],X_level[:, :, 1],)
            tmp_A = A_level[:, :, 2].clone()
            tmp_X = X_level[:, :, 2].clone()
            ParallelScan.compose(A_level[:, :, 1],X_level[:, :, 1],tmp_A,tmp_X,)
            ParallelScan.compose(tmp_A,tmp_X,A_level[:, :, 3],X_level[:, :, 3],)
        elif X_level.size(2) == 2:
            ParallelScan.compose(A_level[:, :, 0],X_level[:, :, 0],A_level[:, :, 1],X_level[:, :, 1],)
            return
        else:  # size == 1
            return

        # DOWNSWEEP
        A_level = A[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        X_level = X[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        ParallelScan.compose(A_level[:, :, 1],X_level[:, :, 1],A_level[:, :, 2],X_level[:, :, 2],)
        for k in range(num_steps - 3, -1, -1):
            A_level = A[:, :, 2**k - 1 : L : 2**k]
            X_level = X[:, :, 2**k - 1 : L : 2**k]
            length = X_level.size(2)
            A_level = A_level.view(B, D, length // 2, 2, -1)
            X_level = X_level.view(B, D, length // 2, 2, -1)
            ParallelScan.compose(A_level[:, :, :-1, 1],X_level[:, :, :-1, 1],A_level[:, :, 1:, 0],X_level[:, :, 1:, 0],)

    @staticmethod
    def scan_reverse(A, X):
        """
        Reverse Blelloch scan used during backward pass. A : (B, D, L, N) X : (B, D, L, N)
        """
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # UPSWEEP (last 2 levels done explicitly)
        A_level = A
        X_level = X
        for _ in range(num_steps-2):
            T = X_level.size(2)
            A_level = A_level.view(B, D, T//2, 2, -1)
            X_level = X_level.view(B, D, T//2, 2, -1)
            X_level[:, :, :, 0].add_(A_level[:, :, :, 0].mul(X_level[:, :, :, 1]))
            A_level[:, :, :, 0].mul_(A_level[:, :, :, 1])
            A_level = A_level[:, :, :, 0]
            X_level = X_level[:, :, :, 0]

        # we have only 4, 2 or 1 nodes left
        if X_level.size(2) == 4:
            X_level[:, :, 2].add_(A_level[:, :, 2].mul(X_level[:, :, 3]))
            A_level[:, :, 2].mul_(A_level[:, :, 3])
            X_level[:, :, 0].add_(A_level[:, :, 0].mul(X_level[:, :, 1].add(A_level[:, :, 1].mul(X_level[:, :, 2]))))
        elif X_level.size(2) == 2:
            X_level[:, :, 0].add_(A_level[:, :, 0].mul(X_level[:, :, 1]))
            return
        else:
            return

        # DOWNSWEEP
        A_level = A[:, :, 0:L:2 ** (num_steps - 2)]
        X_level = X[:, :, 0:L:2 ** (num_steps - 2)]
        X_level[:, :, 1].add_(A_level[:, :, 1].mul(X_level[:, :, 2]))
        A_level[:, :, 1].mul_(A_level[:, :, 2])

        for k in range(num_steps-3, -1, -1):
            A_level = A[:, :, 0:L:2**k]
            X_level = X[:, :, 0:L:2**k]
            length = X_level.size(2)
            A_level = A_level.view(B, D, length//2, 2, -1)
            X_level = X_level.view(B, D, length//2, 2, -1)
            X_level[:, :, :-1, 1].add_(A_level[:, :, :-1, 1].mul(X_level[:, :, 1:, 0]))
            A_level[:, :, :-1, 1].mul_(A_level[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation and saves gradients for backward pass. A,X,H : (B,L,D,N)
        """
        L = X_in.size(1)
        #Cloning is required bcs of in-place operations.
        if L != npo2(L):
            A = pad(A_in) #(B, npo2(L), D, N)
            X = pad(X_in)
        else:
            A = A_in.clone()
            X = X_in.clone()

        A = A.transpose(2,1)# (B, D, npo2(L), N)
        X = X.transpose(2,1)# (B, D, npo2(L), N)
        ParallelScan.scan(A, X) # parallel scan (modifies X in-place)
        H = X.transpose(2,1)[:, :L] # and slice [:, :L] (cut if there was padding
        ctx.save_for_backward(A_in, X) #Saves gradients
        return H
    
    @staticmethod
    def backward(ctx, grad_out):
        """
        Computes the gradients by flowing from output to input. H-> (B,D,L,N) A_in,grad,dA ->(B, L, D, N)
        """
        A_in, H = ctx.saved_tensors
        L = grad_out.size(1)

        # cloning is requiered because of the in-place ops
        if L != npo2(L):
            grad = pad(grad_out) # (B, npo2(L), D, N)
            A_in = pad(A_in) # (B, npo2(L), D, N)
        else:
            grad = grad_out.clone()

        grad = grad.transpose(2,1)
        A_in = A_in.transpose(2,1)  # (B, D, npo2(L), N)

        A_shift = F.pad(A_in[:,:,1:], (0,0,0,1))  # (B, D, npo2(L), N) shift 1 to the left 
        ParallelScan.scan_reverse(A_shift, grad)
        dA = torch.zeros_like(H)
        H_prev = H[:, :, :-1]
        grad_next = grad[:, :, 1:]

        dA = torch.zeros_like(grad)
        dA[:, :, 1:] = H_prev * grad_next
        return (dA.transpose(2,1)[:, :L],grad.transpose(2,1)[:, :L],)
    
pscan = ParallelScan.apply