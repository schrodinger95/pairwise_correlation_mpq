# Test Guide
Guidelines for solving the optimal mixed-precision scheme and testing the performance, BOPs and latency corresponding to this mixed-precision model.

## Getting Started
The folder structures should be the same as following
```
Folder
├──database
├──utils
├──models
```
Important files:
- [database](database): The dataset needed to compute the metrics and solve for the optimal mixed-precision schemes.
- [models](models): Pre-trained 8-bit models.
- [NSGA-II.ipynb](NSGA-II.ipynb): Jupyter notebook that shows how to compute metrics and solve for the optimal mixed-precision schemes.
- [run_test.py](run_test.py): Test code for the entire process of finding the optimal mixed-precision scheme, and testing the accuracy, BOPs, and computation time of the corresponding quantized model.

## Environment
Create the environment from the `environment.yml` file:
```
conda env create -f environment.yml
```
The first line of the yml file sets the new environment's name. 

Activate the new environment: `conda activate myenv`

## Test Command Line
To test the accuracy, BOPs, and latency for the mixed-precision quantized model, use the following command and adjust the attributes accordingly:
```
python3 run_test.py --dataset imagenet \
                    --data /path/to/imagenet/ \
                    --model mobilenetv2_w1 \
                    --limit bops \
                    --limit_p 0.8 \
                    --resume models/mobilenetv2_uniform8/checkpoint.pth.tar
```

## Result

## Credit

The code for the post-training quantization is partly referenced from  [ZeroQ](https://github.com/amirgholami/ZeroQ).