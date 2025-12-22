## Microwave Sky Simulation Suite

This package provides a **microwave sky simulation suite** for generating mock realizations of the **CMB**, **foregrounds**, and **instrumental noise**.

---

### Overview

The core interface is the **`Skysimulator`** class, implemented in `simulator.py`.  
Given a set of user-defined parameters in `config.py`, the simulator generates mock observed microwave sky maps.

Users should modify `config.py` according to their scientific requirements before running the simulation.

---

### Components

#### I. CMB  
(see `cmb_making.py`)

- Unlensed CMB realizations and lensing potential realizations are generated from their input angular power spectra.
- Lensed CMB realizations are obtained by lensing the unlensed CMB with the corresponding lensing potential.
- The lensing operation is performed using **Lenspyx**.

---

#### II. Foregrounds  
(see `foreground_making.py`)

- Foreground templates can be generated using **PySM3**, supporting multiple foreground components.
  - Note that only **one realization** is available for PySM-based templates.
- In addition, parametric models for **Galactic synchrotron** and **thermal dust emission** are provided.
  - Based on these models, **Gaussian realizations** of synchrotron and dust emission can also be generated.

---

#### III. Noise  
(see `noise_making.py`)

- Generates **Gaussian realizations** of instrumental noise, including:
  - **White noise**
  - **1/f noise**
- Noise realizations can be generated either in:
  - **Pixel space**, or
  - **Harmonic space**

---

### Example

An end-to-end example is provided in `generate_sim_example.ipynb`, demonstrating how to:
- Instantiate the **`Skysimulator`** class,
- Generate mock observed maps directly, and
- Generate individual components (CMB, foregrounds, noise) step by step.

---

