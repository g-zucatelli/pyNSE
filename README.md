# pyNSE

This repository mantains Python implementation of objective Non-Stationary Estimators (NSE) applied to Acoustic Non-Stationary Assessment.

For a target acoustic signal $x(t)$ of total length $T$, the non-stationarity estimation is calculated based on local observation window $T_h$ or observation scale $T_h/T \in (0,1]$.

Current available estimators are depicted below:


| Estimator  | ClassID | Reference |
| ------------- | ------------- | - |
| Index of Non-Stationarity  | 'INS'  | <sub>Pierre Borgnat, Patrick Flandrin, Paul Honeine, Cédric Richard, and Jun Xiao, “Testing stationarity with surrogates: A Time-Frequency Approach,” IEEE Transactions on Signal Processing, vol. 58, no. 7, pp. 3459–3470, 2010. <sup>|
| ...  | ...  | <sub> More estimators are on going development... <sup> |


For an objective non-stationary assessment, run the following:
> python calc_ns.py -e `ClassID` -sr `SampleRate` -obs `ObservationScales` -p `PathToFile`

**Example:** Using INS estimator, observation scales $[0.3, 0.4, 0.5]$ and a sample rate of $16$ kHz, the script is given by 

> python calc_ns.py -e 'INS' -sr 16000 -obs 0.3 0.4 0.5 -p audio_file.wav

Objective estimators have been sucessfully adopted in different applications regarding real non-stationary acoustic signals such as speech enhancement [3], synthetic data generation and active learning [2][3].

Additional parse options can be assessed at `calc_ns.py`. 
Further references are also provided below for consultation.

This repository is under the GNU General Public License and the code is entirely based on referenced authors.

---
### References
[1] G. Zucatelli and R. Coelho, “Adaptive learning with surrogate assisted training models using limited labeled acoustic sample sequences”. 2021 IEEE Statistical Signal Processing Workshop (SSP). IEEE, 2021, pp. 21–25

[2] G. Zucatelli and R. Coelho, “Adaptive reverberation absorption using non-stationary masking components detection for intelligibility improvement”. IEEE Signal Processing Letters 27, 2019: pp. 1-5.

[3] G. Zucatelli, R. Coelho, and L. Zão, “Adaptive learning with surrogate assisted training models for acoustic source classification”. IEEE Sensors Letters, vol. 3, no. 6, pp. 1–4, 2019.

[4] Pierre Borgnat, Patrick Flandrin, Paul Honeine, Cédric Richard, and Jun Xiao, “Testing stationarity with surrogates: A Time-Frequency Approach”. IEEE Transactions on Signal Processing, vol. 58, no. 7, pp. 3459–3470, 2010.