# QIC project: Quantum PCA
End-of-Semester project of the course *Quantum Information and Computing* at Università degli studi di Padova, A.Y. 2024/2025. 

Contributors:
- Laura Schulze
- Mila Šuput

In this project, we aim to implement **Quantum PCA** using `qiskit`. We then use both classical and quantum PCA to analyse the MNIST handwritten digit dataset.

It contains the following files and folders:
- **qiskit_playground.ipynb**: Jupyter Notebook used to familiarize ourselves with qiskit and understand the different steps of qPCA.
- **qPCA_clean.ipynb**: Notebook containing the proper implementations of the different functions for qPCA. Used to adjust different parameters and play around with different test matrices.
- **dens_exp.py**: Contains helper functions to construct the unitary operator for the qPCA Phase estimation using the Swap gate method described in [this paper](https://doi.org/10.1038/nphys3029).
