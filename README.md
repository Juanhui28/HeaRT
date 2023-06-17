# HeaRT

Official code for the paper "Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking".


## Installation

The experiments were run using python 3.9. The required packages and their versions can be installed via the `requirements.txt` file. 
```
pip install -r requirements.txt
``` 

One exception is the original PEG code (`benchmarking/baseline_models/PEG`) were run using python 3.7 and the packages in the `peg_requirements.txt` file.


## Download Data

The negative samples for the new evaluation setting can be downloaded via:
```
curl https://cse.msu.edu/~shomerha/HeaRT-Data/dataset.tar.gz | tar -xvz
``` 
This will create a directory called `dataset`. Please ensure that this directory is placed in the root project directory.


## Reproduce Results

To reproduce the results, please refer to the settings in **hyparameter** directory


## Cite



