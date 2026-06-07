
import itertools
import subprocess
import csv
import os


weight_decays = [ 0.01, 0.05, 0.1, 0.0]
dropouts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.0]
architectures = [
    {"d_model": 128, "n_layer": 2, "d_state": 16},
    {"d_model": 128, "n_layer": 4, "d_state": 16},
    {"d_model": 128, "n_layer": 6, "d_state": 16},
    {"d_model": 256, "n_layer": 2, "d_state": 16},
    {"d_model": 256, "n_layer": 4, "d_state": 16},
    {"d_model": 256, "n_layer": 6, "d_state": 16},
    {"d_model": 64,  "n_layer": 2, "d_state": 32},
    {"d_model": 64,  "n_layer": 4, "d_state": 32},
    {"d_model": 64,  "n_layer": 6, "d_state": 32},
    {"d_model": 128, "n_layer": 2, "d_state": 32},
    {"d_model": 128, "n_layer": 4, "d_state": 32},
    {"d_model": 128, "n_layer": 6, "d_state": 32},
    {"d_model": 256, "n_layer": 2, "d_state": 32},
    {"d_model": 256, "n_layer": 4, "d_state": 32},
    {"d_model": 256, "n_layer": 6, "d_state": 32},
    {"d_model": 64,  "n_layer": 6, "d_state": 16},
]
schedulers = ["cosine"]
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)
all_experiments = list(itertools.product(schedulers,weight_decays,dropouts,architectures,))
total_runs = len(all_experiments)

with open("experiments.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["exp_id","d_model","n_layer","d_state","dropout","weight_decay","scheduler",])
    for run_num, ( scheduler, wd,dropout, arch) in enumerate(all_experiments, start=1):
        exp_id = f"exp_{run_num:02d}"
        writer.writerow([exp_id,arch["d_model"],arch["n_layer"],arch["d_state"],dropout,wd,scheduler,])
        cmd = ["python3","train.py","--d_model", str(arch["d_model"]),"--n_layer", str(arch["n_layer"]),
               "--d_state", str(arch["d_state"]),"--dropout", str(dropout),"--weight_decay", str(wd),
               "--scheduler", scheduler,  "--savefilename", f"checkpoints/{exp_id}.pt","--logfilename", f"logs/{exp_id}.log",]
        print(f"[{run_num}/{total_runs}] {exp_id}")
        subprocess.run(cmd, check=True)