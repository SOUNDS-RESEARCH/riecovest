# Riemannian covariance estimation (riecovest)
This is a package for estimation of signal covariance matrices from noisy signal and noise-only data. The package is using pymanopt to perform optimization over Riemannian manifolds, using JAX to as a backend for automatic differentiation.

**[More info and complete API documentation](https://sounds-research.github.io/riecovest/)**

## References
The code was developed as part of the paper [J. Brunnström, M. Moonen, and F. Elvander, “Robust signal and noise covariance matrix estimation using Riemannian optimization,” presented at the European Signal Processing Conference (EUSIPCO), Sep. 2024](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000291.pdf). The examples folder contains code to replicate the results from the paper. If you use this software in your research, please cite the paper. 
```
@inproceedings{brunnstromRobust2024,
  title = {Robust Signal and Noise Covariance Matrix Estimation Using {{Riemannian}} Optimization},
  booktitle = {European {{Signal Processing Conference}} ({{EUSIPCO}})},
  author = {Brunnstr{\"o}m, Jesper and Moonen, Marc and Elvander, Filip},
  year = {2024},
  month = sep,
  keywords = {covariance estimation,GEVD,low-rank,manifolds,MWF,quotient manifold,riemannian optimization,robust,SPD}
}
```

## License
The software is distributed under the MIT license. See the LICENSE file for more information.

## Installation
The package can be installed via pip from the PyPi repository:
```
pip install riecovest
```
Alternatively, clone this repository and install the package using:
```
pip install path/to/riecovest
```

To run the examples "reproduce_covest_t_distribution.py" and "reproduce_spatial_filtering.py" which reproduces the results from the aforementioned paper, a longer list of dependencies is needed. These can be installed with pip using:
```
pip install riecovest[examples]
```
The example "reproduce_spatial_filtering.py" also makes use of the [MeshRIR](https://www.sh01.org/MeshRIR/) dataset. It must be downloaded from the original source along with the dataset loader irutilities.py. The dataset loader must be placed in the same folder as the example script.

## Acknowledgements
The software has been developed during a PhD project as part of the [SOUNDS ETN](https://www.sounds-etn.eu) at KU Leuven. The SOUNDS project has recieved funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 956369.
