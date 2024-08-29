This folder perform CMB delensing with Gradient template method, which generate a lensing QU template map (therefore lensing B-mode template map).
We can either subtract it from the observation or add it to the likelihood as a pseudo channel map.


### Use the combined tracer to delensing
1.temp_forecast_comb.py : Use the combined tracer to delensing.
2.post_cal_temp_comb.py : Calculate the auto- and cross- spectra between the observed B-mode and the B-mode template.

### Use the internal reconstructed phi to delensing
1.temp_forecast_rec_new_format.py : Use the internal reconstructed phi to delensing
2.post_cal_temp_rec_new3.py : Calculate the auto- and cross- spectra between the observed B-mode and the B-mode template.
3.temp_forecast_rec_debias.py : Use a Gaussian phi noise realization draw from N0 plus phi signal, instead of using the internal reconstructed phi.
Since they have the same total power, while some bias terms cancel in the former case, so we can different the two case to obtain these bias terms. 