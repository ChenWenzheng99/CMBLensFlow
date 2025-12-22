# ✨✨✨ Lensflow ✨✨✨

![image](https://github.com/ChenWenzheng99/We-love-lensing/blob/main/image/deflection.png)  
*(Image credit: © ESA)*  


**Lensflow** is a comprehensive package designed to provide **useful tools and reference workflows for CMB B-mode delensing forecasts**.

Parts of this codebase have been used in the following studies:

- **Delensing for Precision Cosmology: Optimizing Future CMB B-mode Surveys to Constrain r**  
  https://arxiv.org/abs/2502.18790  
  *In this work, we consider parameterized Galactic foreground models and external delensing.*

- **From South to North: Leveraging Ground-Based LATs for Full-Sky CMB Delensing and Constraints on r**  
  https://arxiv.org/abs/2507.19897  
  *This study emphasizes the critical role of large-aperture telescopes located in the Northern Hemisphere for CMB B-mode delensing and constraints on* \( r \).

---

## Package Overview

This package consists of the following main components:

**Ⅰ. Microwave sky simulation suite**  
Simulation of CMB, foregrounds, and instrumental noise.

**Ⅱ. Internal Linear Combination (ILC) suite**  
ILC performed in multiple domains (pixel, harmonic, Fourier, and needlet), with optional component deprojection.

**Ⅲ. Lensing reconstruction framework**  
- Internal CMB lensing reconstruction using **Plancklens**  
- External lensing reconstruction using LSS tracers (e.g., CIB, galaxy number density fluctuations), based on two-point statistics

**Ⅳ. CMB B-mode delensing pipeline**  
SBATCH-based simulation workflows implementing two delensing methods:
- Gradient-order template method  
- Inverse-lensing method  

Auto- and cross-power spectra between observed B-modes and lensing B-mode templates are computed.

**Ⅴ. Parameter constraint example**  
End-to-end parameter inference using **Cobaya**.

---

## Dependencies

### Ⅰ. Lensing Reconstruction
- **Plancklens**  
  https://github.com/carronj/plancklens  
- **Lenspyx 2.0.0** (for `lens` environment)  
  https://github.com/carronj/lenspyx/releases/tag/v2.0.0  
- **Healpy**  
  https://github.com/healpy/healpy  
- **NumPy**  
- **Matplotlib / PyLab**  
- **mpi4py** (optional)

### Ⅱ. Foreground Simulation
- **Astropy**  
  https://github.com/astropy/astropy  
- **PySM3**  
  https://pysm3.readthedocs.io/en/latest/index.html#installation  
- **PyMaster (NaMaster)**  
  https://namaster.readthedocs.io/en/latest/source/installation.html  

### Ⅲ. CMB B-mode Delensing
- **CMBlensplus**  
  https://github.com/toshiyan/cmblensplus  
- **SciPy**

### Ⅳ. Parameter Constraints
- **Cobaya**  
  https://github.com/CobayaSampler/cobaya  
- **CAMB**  
  https://github.com/cmbant/CAMB  

---

## CMB Delensing Pipeline

A schematic overview of the CMB delensing workflow implemented in this package is shown below:

![image](https://github.com/ChenWenzheng99/Lensing-tracers/blob/main/image/pipeline.png)

---

This package is intended for **simulation-based forecasts**, methodological development, and end-to-end validation of **CMB B-mode delensing strategies**.

---

## Notes and Contributions

Given the limitations of the author’s experience and available resources, **any kind suggestions, feedback, or corrections are sincerely welcomed**.  
This project aims to serve as a collaborative effort toward building a more robust and practical **CMB B-mode delensing workflow**.

