import sys, os
sys.path.insert(0, '../params')
import params as parfile
from library_parameter import *
import healpy as hp
import numpy as np
from plancklens import utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import mpi

_CONFIG_FILE = os.environ.get("ALI_CONFIG", "config.yaml")
with open(_CONFIG_FILE, "r") as f:
    _cfg = yaml.safe_load(f)


# ===== Parameters =====
qe_key = _cfg['qe']["lib_qe_keys"][0]                # e.g., 'ptt', 'ptt_bh', 'p_p', 'p', etc.

_rcfg = _cfg["reconstruction"]          

if _rcfg.get("nsims_from_seeds", True):
    num_sims = len(seeds) - 1               # The number of simulations to calculate qlm reconstruction for
else:
    num_sims = _rcfg["nsims"]         


# ===== Check MPI =====
print(f"[Process PID {os.getpid()}] Rank={mpi.rank}, Size={mpi.size}")

# ===== Task Division =====
all_tasks = list(range(num_sims))
my_tasks = all_tasks[mpi.rank::mpi.size]
print(f"[Rank {mpi.rank}] Assigned tasks: {my_tasks}")

# ===== Main Loop =====
for sindex in my_tasks:
    try:
        print(f"[Rank {mpi.rank}] Start sindex={sindex}")
        
        qlm = parfile.qlms_dd.get_sim_qlm(qe_key, sindex)

    except Exception as e:
        print(f"[Rank {mpi.rank}] Error on sindex={sindex}: {e}")

# ===== Synchronize =====
mpi.barrier()
if mpi.rank == 0:
    print("✅ All qlm reconstructions finished successfully.")
