.. aspcol documentation master file, created by
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
The software is distributed under the MIT license. See the LICENSE file for more information. If you use this software in your research, please cite the following paper

Acknowledgements
================
The software has been developed during a PhD project as part of the `SOUNDS <https://www.sounds-etn.eu/>`_ ETN project at KU Leuven. The SOUNDS project has recieved funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 956369.

