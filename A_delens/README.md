This repository provides a **CMB B-mode delensing forecast suite**, implementing both the **gradient-order template method** and the **inverse-lensing method**.

---

## Overview

The scripts **`sbatch_gradient_template`** and **`sbatch_inverse_lensing`** offer **user-friendly delensing pipelines**, which include:

- Automatic loading of observed CMB maps and lensing proxies, followed by **Wiener filtering**
- Construction of **lensing B-mode templates** using the two delensing approaches
- Computation of:
  - Auto- and cross-power spectra of the observed B-modes and the lensing B-mode templates  
  - Bias terms and corresponding scale factors

---

## How to Run

To run these scripts, you only need to:
1. Modify the **input data directories** specified at the beginning of each script  
2. Set the relevant **configuration parameters** at the end of the scripts  

---

## Important Notes

- These pipelines are intended **solely for simulation-based forecasts**.  
- We assume **perfect knowledge of residuals** for debiasing purposes.  
- For **real data analyses**, additional treatments—such as residual modeling, uncertainty propagation, and systematic-error mitigation—would be required and are **not included** in the current implementation.

---

This suite is designed for fast, transparent evaluation of **delensing performance** under idealized forecast assumptions.
