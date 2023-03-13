# pyINS

This repository mantains a Python implementation of the Index of Non-Stationary (INS) [1].

---
## Theoretical Overview
For a target signal $x(t)$, the INS is obtained considering its multitaper spectral representation $S_x(l,f)$ as

$$
  S_x(l,f) = \frac{1}{K} \sum_{k} S_x^{(h_k)} (l,f),
$$


where $l$ is the frame, $f$ is the frequency bin and  $S_x^{(h_k)} (l,f)$ is the spectrogram obtained considering the $k$-th Hermitian function $h_k(t)$ as the taper. 
In other words, its is possible to represent $S_x^{(h_k)} (l,f)$ as

$$
  S_x^{(h_k)} (l,f) = \left| \int x(s) h_k(s-l) e^{-j 2\pi fs} \right|^2 
$$

for $h_k(t) = e^{-t^2/2} H_k(t) / \sqrt{\pi^{1/2} 2^k k!}$,
where $H_k(t)$ are Hermite polynomials obtained by recursion as

$$
H_k(t) = 2t H_{k-1}(t) - 2(k-2)H_{k-2}(t),
$$

for $k \ge 2$ and initializations of $H_0(t)=1$ and $H_1(t)=2t$. 

This measure compares the target signal with stationary references called surrogates, adopting the symmetric Kullback-Leibler distance and log-spectral deviation.

Surrogate signals are generated by changing the phase of the spectral representation of $x(t)$ to realizations of a uniform distribution $\mathcal{U}[-\pi,\pi]$, which then guarantees their stationary behavior [1].

The comparison is carried out for different time scales $T_h/T$, where $T_h$ is the short-time spectral analysis length and $T$ is the total signal duration.
For each length $T_h$, a threshold $\gamma \approx 1$ is defined to keep the stationarity assumption considering a $95\%$ confidence degree as

- INS $\leq \gamma$ → signal is stationary
- INS > $\gamma$ → signal is non-stationary


As an example, consider a Dog Bark acoustic source from the UrbanSound database [2]:
<div align="center">
  <picture>
    <img alt="Spectrogram" width="50%" align="center" src="https://www.linkpicture.com/q/dogs_spec_tight.jpeg">
  </picture>
  <picture>
    <img alt="INS" width="50%" align="center" src="https://www.linkpicture.com/q/dogs_ins_tight_1.jpeg">
  </picture>
</div>

For this example, the signal attains a maximum INS value above the non-stationary threshold ($\gamma$), which means it presents a non-stationary behavior.

This repository is under the GNU General Public License and the code is entirely based on referenced authors personal material.

---
### References
[1] Pierre Borgnat, Patrick Flandrin, Paul Honeine, Cédric Richard, and Jun Xiao, “Testing stationarity with surrogates: A Time-Frequency Approach,” IEEE Transactions on Signal Processing, vol. 58, no. 7, pp. 3459–3470, 2010.

[2] Justin Salamon, Christopher Jacoby, and Juan Pablo Bello, “A dataset and taxonomy for urban sound research,” in Proceedings of the 22nd ACM international conference on Multimedia, 2014, pp. 1041–1044.
