This repository provides a **user-friendly example template** for performing **internal CMB lensing reconstruction** using **Plancklens**.

---

## Quick Start

**Ⅰ.** Modify the configuration file **`config.yaml`** according to your scientific and technical requirements.

**Ⅱ.** Refer to **`one.sh`** for an example workflow that runs the reconstruction pipeline and produces the reconstructed **QLMs** and corresponding **power spectra (Cl)**.

---

## Key Files and Their Roles

Below is a brief overview of important scripts and configuration files in this template:

- **`ali2020_sims.py`**  
  Defines functions for loading simulated or observed sky maps.

- **`delete_hash.sh`**  
  Removes hash files in the working directory while preserving all reconstructed intermediate results.  
  This allows you to **resume or continue runs** after modifying the configuration midway.

- **`get_ninv.py`**  
  Computes the inverse-noise weighting (`N⁻¹`) from the provided noise RMS (nrms) map.

- **`lensingrec_quick.py`**  
  Main script for running lensing reconstruction to obtain reconstructed QLMs, noise power spectra, etc.

- **`lensingrec_quick_mpi.py`**  
  MPI-parallelized version of the reconstruction pipeline for large-scale or high-resolution runs.

- **`lensingrec_quickstart_red.ipynb`**  
  Jupyter notebook for quick inspection and validation of reconstruction results.

- **`library_parameter.py`**  
  A container for global configuration and shared parameters.

- **`one.py`**  
  Additional parameter container used by the reconstruction pipeline.

- **`parameter.py`**  
  Defines instantiated parameter objects used during execution.

- **`plot.py`**  
  Visualization utilities for reconstructed lensing power spectra and covariance matrices.

- **`submit_sbatch.sh`**, **`submit_sbatch_mpi.sh`**  
  SLURM submission scripts for running the reconstruction on HPC clusters, with or without MPI.

---

This template is designed to be modular, restart-friendly, and suitable for both **single-node** and **MPI-parallel** lensing reconstruction workflows.
