# Installation

We detail the installation process using Conda on Linux.

We note that the installation can be done via Pip. However, we find it's easier and more reliable with Conda.

## 1. Install Conda
```
curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

## 2. Setup Code and Packages

The system and CUDA version used are:
- Ubuntu 20.04.6 LTS
- CUDA 11.6
- Python 3.9

We first clone the repository.
```
git clone git@github.com:Juanhui28/HeaRT.git
cd HeaRT/
```

The requirements can be found in the `heart_env.yml` file. Installing this will also create an environement for the project, `heart_env`. 
```
# Install environment requirements
conda env create -f heart_env.yml   

# Activate environment
conda activate heart_env
```
**If you have encounter any difficulties using conda , we suggest installing the key package PyG first. You can install other missing packages from the error messages by running codes. The offical PyG installation guide is [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).**  For your reference, our current environment includes:

- python: 3.9
- torch: 1.12.1
- torch cuda: 11.6
according to the installation guide of pyg, we can use the following command:
```
TORCH=1.12.1
CUDA=cu116
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```


The one exception is when running the PEG model. As oppossed to other methods it utilizes:
- CUDA 10.2
- Python 3.7

The environment for PEG can be installed via:
```
# Install environment requirements
conda env create -f peg_env.yml   

# Activate environment
conda activate peg_env
```


We note that the correct CUDA version can be installed multiple ways. For manual installation, please see [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). On a HPCC system, the correct version can be ativated via SLURM. Please see [here](https://hpcf.umbc.edu/gpu/how-to-run-on-the-gpus/) for more details.
