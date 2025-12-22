## ILC Component Separation Suite

This package provides an **ILC (Internal Linear Combination) suite** for component separation in CMB data analysis.

The ILC algorithm can be implemented in several domains:
- **Pixel space** (`pilc.py`)
- **Harmonic space** (`hilc.py`)
- **Fourier space** (`filc.py`)
- **Needlet space** (`nilc.py`)

The **NILC** implementation is adapted from  
<https://github.com/dreamthreebs/openilc>, where detailed usage instructions are available.

Because standard ILC methods can be inefficient at suppressing certain **localized foreground residuals**—particularly **extragalactic point sources**, the **thermal Sunyaev–Zel’dovich (tSZ) effect**, and similar contaminants—the suite also supports **constrained ILC** techniques. These allow specific foreground components to be explicitly deprojected during the ILC optimization.

In practice, we recommend using **HILC** or **NILC**, as they consistently outperform pixel- and Fourier-based implementations. A commonly adopted and effective strategy is:
- **NILC** for cleaning **SAT-observed B-mode maps**, and
- **HILC** for cleaning **LAT-observed T/E/B maps**.

An example of a full-sky **HILC implementation for temperature data**, including the deprojection of multiple extragalactic foreground components, is provided in  
`test_hilc_full_T.ipynb`.
