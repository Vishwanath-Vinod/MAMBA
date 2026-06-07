import argparse
import torch
from torch.optim.lr_scheduler import (LambdaLR,StepLR,CosineAnnealingLR,)

def get_device_from_arg(device_id):
    if (device_id is not None and torch.cuda.is_available() and 0 <= device_id < torch.cuda.device_count()):
        print(f"cuda:{device_id}")
        return torch.device(f'cuda:{device_id}')
        
    else:
        print("CPU")
        return torch.device('cpu')

def make_train_parser():
    """
    This parses arguments for the particular training  run.
    """
    parser = argparse.ArgumentParser(description="Train MambaLM on WikiText-2")

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--optimizer",type=str,default="adamw",choices=["sgd", "adam", "adamw"],)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--scheduler",type=str,default="step",choices=["constant", "step", "cosine"],)
    # StepLR parameters
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    # Cosine parameters
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--savefilename', type=str, default='./saved_models/model1.pt')
    parser.add_argument('--logfilename', type=str, default='./logs/out1')

    parser.add_argument('--data_dir', type=str, default='./data/wikitext-2',help="Path to WikiText-2 directory")
    parser.add_argument('--context_size', type=int, default=150)
    parser.add_argument('--max_vocab_size', type=int, default=-1)
    return parser


def build_scheduler(args, optimizer):
    """
    constant : fixed learning rate
    step     : lr *= gamma every step_size epochs
    cosine   : cosine annealing to eta_min
    """
    if args.scheduler == "constant":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer,step_size=args.step_size,gamma=args.gamma,)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer,T_max=args.num_epochs,eta_min=args.min_lr,)
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    return scheduler
           
def setup_optimizer_from_args(args, model):
    """
    Setsup the Optimizer using inputs from arguments passed during training.
    """
    lr = args.lr
    print('Using lr =', lr)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay,)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=args.weight_decay,)
    return optimizer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds):
    hours, rem = divmod(int(seconds), 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
