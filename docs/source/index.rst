.. riecovest documentation master file, created by
   sphinx-quickstart on Wed Sep 13 10:22:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Riemannian covariance estimation (riecovest)
============================================
This is a package for estimation of signal covariance matrices from noisy signal and noise-only data. The package is using pymanopt to perform optimization over the specified Riemannian manifolds.

API Reference
=============
.. autosummary::
   :toctree: _autosummary
   :template: new-module-template.rst

   riecovest.distance
   riecovest.matrix_operations
   riecovest.covariance_estimation
   riecovest.random_matrices

License
=======
The software is distributed under the MIT license. See the LICENSE file for more information. 

The package was developed in the course of the following research. Please consider citing the following papers if relevant to your work. 

*Robust signal and noise covariance matrix estimation using Riemannian optimization*, J. Brunnström, M. Moonen, and F. Elvander

.. code-block:: bibtex

   @inproceedings{brunnstromRobust2024,
   title = {Robust Signal and Noise Covariance Matrix Estimation Using {{Riemannian}} Optimization},
   booktitle = {European {{Signal Processing Conference}} ({{EUSIPCO}})},
   author = {Brunnstr{\"o}m, Jesper and Moonen, Marc and Elvander, Filip},
   year = {2024},
   month = sep,
   pages={291-295},
   keywords={Manifolds;Heavily-tailed distribution;Noise;Estimation;Europe;Euclidean distance;Cost function;Eigenvalues and eigenfunctions;Covariance matrices;Synthetic data;noise reduction;robust covariance matrix estimation;Riemannian optimization;Hermitian positive matrices;manifolds},
   doi={10.23919/EUSIPCO63174.2024.10715247}}
   }

Acknowledgements
================
The software has been developed during a PhD project as part of the `SOUNDS <https://www.sounds-etn.eu/>`_ ETN project at KU Leuven. The SOUNDS project has recieved funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 956369.

