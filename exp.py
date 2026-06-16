
import itertools
import subprocess
import csv
import os
weight_decays = [0.01]
dropouts = [0.0,0.1,0.2,0.3,0.4,0.5]
vocab_sizes = [1000, 5000, 10000, -1]

architectures = [
    {"d_model": 256, "n_layer": 6, "d_state": 16},
    {"d_model": 128, "n_layer": 4, "d_state": 16},
    {"d_model": 64,  "n_layer": 2, "d_state": 16},
]

scheduler = "cosine"
os.makedirs("experiments", exist_ok=True)
exp_num = 1
for i, arch in enumerate(architectures):
    arch_dir = os.path.join("experiments", str(i))
    os.makedirs(arch_dir, exist_ok=True)
    ckpt_dir = os.path.join(arch_dir, "checkpoints")
    log_dir = os.path.join(arch_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    summary_file = os.path.join(arch_dir, "summary.txt")
    if not os.path.exists(summary_file):
        with open(summary_file, "w") as f:
            f.write(f"Architecture\n"
                f"------------\n"
                f"d_model = {arch['d_model']}\n"
                f"n_layer = {arch['n_layer']}\n"
                f"d_state = {arch['d_state']}\n\n"
            )
            f.write("Experiments\n")
            f.write("-----------\n")

    for vocab_size, dropout, wd in itertools.product(vocab_sizes,dropouts,weight_decays,):
        exp_id = f"exp_{exp_num:03d}"
        with open(summary_file, "a") as f:
            f.write(f"Experiment{exp_id} : "f"vocabulary size={vocab_size} "f"dropout={dropout} "f"weight decay={wd}\n")
        savefile = os.path.join(ckpt_dir,f"{exp_id}.pt",)
        logfile = os.path.join(log_dir,f"{exp_id}.log",)
        cmd = ["python3","train.py","--d_model",str(arch["d_model"]),"--n_layer",str(arch["n_layer"]),"--d_state",str(arch["d_state"]),
               "--max_vocab_size",str(vocab_size),"--dropout",str(dropout),"--weight_decay",str(wd),"--scheduler",scheduler,
               "--savefilename",savefile,"--logfilename",logfile,]
        subprocess.run(cmd, check=True)
        exp_num += 1