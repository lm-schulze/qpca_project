# QIC project: Quantum PCA
End-of-Semester project of the course *Quantum Information and Computing* at Università degli studi di Padova, A.Y. 2024/2025. 

Contributors:
- Laura Schulze ([@lm-schulze](https://github.com/lm-schulze))
- Mila Šuput ([@msuput](https://github.com/msuput))

In this project, we aim to implement **Quantum PCA** using `qiskit`. Inspired by [this paper](https://doi.org/10.1109/QCE57702.2023.10175) and the [corresponding github repository](https://github.com/Eagle-quantum/QuPCA).

It contains the following files and folders:
- **qPCA_clean.ipynb**: Notebook containing the proper implementations of the different functions for qPCA. Used to adjust different parameters and play around with different test matrices.
- **dens_exp.py**: Contains helper functions to construct the unitary operator for the qPCA Phase estimation using the Swap gate method described in [this paper](https://doi.org/10.1038/nphys3029).
- **qPCA_funcs.py**: Collection of helper functions from qPCA_clean.ipynb to be used in the MNIST application.
- **DME_poc.ipynb**: Notebook containing the proof of concept of the density matrix exponentiation through simulation on denisty matrices.
- **qPCA_plots.ipynb**: Notebook for comparing accuracy for different resolutions and with some circuit details.

