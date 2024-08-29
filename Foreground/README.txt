This folder contains the simulation set of galactic foreground and instrumental white noise.
This galactic foreground focus on the dust and synchrotron which are the major contamination to CMB polarization observation.

1.We use a seven-parameter model to calculate their power, and draw realization respectively. 
For a single simulation set, the seed of each components is fixed, and we just scale the intensity of one of the frequency map to other frequencies. The final foreground map is the sum of each component maps.
Notice that the parameter is fitted from the power spectra of the template map from PySM3(which actually based on Planck and WMAP observation maps), and the mask corresponds to AliCPT 10% sky patch.

2.The instrumental white noise is designed for the SAT and LAT for each frequencies, these noise realization will be used for CMB B-mode delensing and parameter constraint.
