# Test Guide
## File Tree
- [database](database): The dataset needed to compute the metrics and solve for the optimal mixed-precision schemes.
- [models](models): Pre-trained 8-bit models.
- [NSGA-II.ipynb](NSGA-II.ipynb): Jupyter notebook that shows how to compute metrics and solve for the optimal mixed-precision schemes.
- [run_test.py](run_test.py): Test code for the entire process of finding the optimal mixed-precision scheme, and testing the accuracy, BOPs, and computation time of the corresponding quantized model.
```
Folder
├──database
├──utils
├──models
```

## Environment
Create the environment from the `environment.yml` file:
```
conda env create -f environment.yml
```
The first line of the yml file sets the new environment's name. 

Activate the new environment: `conda activate myenv`

## Test Command Line
```
python
```