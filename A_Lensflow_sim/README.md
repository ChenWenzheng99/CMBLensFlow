This is a microwave sky simulation suite, generate realization of CMB, foreground and noise.

Summary:

A class **Skysimulator** can be found in **simulator.py**, which generate mock observation given the setting parameters in **config.py**. Modify the **config.py** according to your requirement.

Ⅰ. CMB: see **cmb_making.py**. Unlensed CMB and lensing potential realizations are generated based on angular power spectra, lensed CMB realizations are obtained by lensing the unlensed CMB with lensing potential, performed with **Lenspyx**.

Ⅱ. Foreground: see **foreground_making.py**. A generator based on **PySM3** can be used to generate multiple component templates, notice that only one realization available. 
            Besides, parameter models of Galactic synchrotron and dust emission is provided, one can also generate Guassian Galactic synchrotron and dust emission realization based on the models.

Ⅲ. Noise: see **noise_making.py**. Generate Guassian realizations of 1/f + white noise, either from pixel domain or harmonic domain.

An example can be found in **generate_sim_example.ipynb**, where we call class **Skysimulator** to generate mock observation. We also generate each components step-by-step.
