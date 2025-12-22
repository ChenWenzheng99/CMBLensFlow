This is an ILC suite used for component separation. 

ILC can be performed in pixel, harmonic, Fourier, needlet domain. See **pilc.py**, **hilc.py**, **filc.py**, **nilc.py**, respectively. 
  (NILC part is developed by https://github.com/dreamthreebs/openilc, find detailed instrument there.)

Besides, since the ILC method perform poor in mitigating some point-like residual from foreground (especially extragalactic point source, tSZ, etc.), one can deproject out certain components in the ILC procedure (i.e. constrained ILC).

See **test_hilc_full_T.ipynb** for an example of T-field HILC implementation, with different extragalatic foreground deprojection.
