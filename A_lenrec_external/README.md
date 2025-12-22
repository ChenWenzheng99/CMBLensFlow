This repository provides a **simulation suite for large-scale structure (LSS) tracers**, designed for **forecast studies of CMB external delensing**.

---

## Overview

The current simulations are based on **two-point statistics (2-pt functions)** only. As a result, they **do not include primordial non-Gaussianity (PNG)**, which may be important for detailed and realistic data analyses.  
For more accurate and comprehensive modeling, **N-body–based simulations** may be required in future extensions.

---

## Key Files

- **`get_LSS_power5_example.ipynb`**  
  An example and reference workflow for computing **theoretical angular power spectra**, including:
  - CMB lensing convergence  
  - Cosmic Infrared Background (CIB)  
  - Galaxy number density fluctuations  

  Both **imaging surveys** (e.g., *Euclid*, *LSST*) and **spectroscopic surveys** (e.g., *MUST*) are considered.

- **`run_LSS_ones_MUST_example.ipynb`**  
  An example demonstrating how to run the LSS tracer simulations using the theoretical power spectra computed above.  
  The outputs include:
  - Individual LSS tracer realizations  
  - An **optimally combined tracer**, constructed to **maximize CMB delensing efficiency**

---

This suite is intended for **forecast-level studies** and provides a flexible framework for testing the delensing performance of different LSS tracers and survey strategies.
