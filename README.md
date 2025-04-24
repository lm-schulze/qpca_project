# QIC project: Quantum PCA
End-of-Semester project of the course *Quantum Information and Computing* at Università degli studi di Padova, A.Y. 2024/2025. 

Contributors:
- Laura Schulze
- Mila Šuput

In this project, we aim to implement **Quantum PCA** using `qiskit`. We then use both classical and quantum PCA to analyse the MNIST handwritten digit dataset.

It contains the following files and folders:
- **qiskit_playground.ipynb**: Jupyter Notebook used to familiarize ourselves with qiskit and understand the different steps of qPCA. Inspired by [this paper](https://doi.org/10.1109/QCE57702.2023.10175) and the [corresponding github repository](https://github.com/Eagle-quantum/QuPCA).
- **qPCA_clean.ipynb**: Notebook containing the proper implementations of the different functions for qPCA. Used to adjust different parameters and play around with different test matrices.
- **dens_exp.py**: Contains helper functions to construct the unitary operator for the qPCA Phase estimation using the Swap gate method described in [this paper](https://doi.org/10.1038/nphys3029).
- **qPCA_attempt.ipynb**: An attempt to integrate the density matrix exponentiation from dens_exp.py into the qPCA implementation from qPCA_clean.ipynb. Currently not functional.
- **qPCA_funcs.py**: Collection of helper functions from qPCA_clean.ipynb to be used in the MNIST application.
- **MNIST_qPCA.ipynb**: Notebook trying to apply qPCA to the MNIST handwritten digits dataset. WIP.
- **cov_matr_preparation.ipynb**: Attempt to implement the Data Loading via ensemble average density matrix, as suggested [here](https://doi.org/10.1103/PRXQuantum.3.030334). WIP.


