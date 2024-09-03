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
Important codes:
- [NSGA-II.ipynb](NSGA-II.ipynb): Jupyter notebook that shows how to compute metrics and solve for the optimal mixed-precision schemes.
- [run_test.py](run_test.py): Test code for the entire process of finding the optimal mixed-precision scheme, and testing the accuracy, BOPs, and latency of the corresponding quantized model.

## Environment
Create the environment from the `environment.yml` file:
```
conda env create -f environment.yml
```
The first line of the yml file sets the new environment's name. 

Activate the new environment: `conda activate pairwise`

## Test Command Line
To test the accuracy, BOPs, and latency for the mixed-precision quantized model, use the following command and adjust the attributes accordingly:
```
python3 run_test.py --dataset imagenet \
                    --data /path/to/imagenet/ \
                    --model mobilenetv2_w1 \
                    --limit bops \
                    --limit_p 0.6 \
                    --resume models/mobilenetv2_uniform8/checkpoint.pth.tar
```
- `dataset`: `imagenet` or `cifar10`.
- `data`: Path to dataset.
- `model`: `mobilenetv2_w1` or `resnet50`.
- `limit`: `size` or `bops`.
- `limit_p`: Percentile between 4-bit and 8-bit models, should be between (0, 1).
- `resume`: Path to the checkpoint (default: none).

The test takes approximately 10 minutes on NVIDIA RTX 4090.

## Result
| Model       | Bit-Width | Accuracy (%) | BOPs (G) | Latency (s) |
|-------------|-----------|--------------|----------|-------------|
| MobileNetV2 | 8-bit     | 72.59        | 41.03    | 5.69e-03    |
| MobileNetV2 | MP        | 71.71        | 28.77    | 4.49e-03    |
| ResNet50    | 8-bit     | 77.60        | 246.44   | 8.07e-03    |
| ResNet50    | MP        | 76.47        | 149.17   | 6.05e-03    |

`Accuracy` is the performance of the model with respect to the `imagenet`, `BOPs` is the number of bit multiplication operations performed by the model during forward inference, and `latency` is the time taken by the model to inference on an input.

To get the result for `MobieNetV2`, set `limit` to `bops` and `limit_p` to `0.6`. Accuracy is reduced by 0.88%, BOPs is improved by 29.87% and latency is improved by 21.02%.
```
python3 run_test.py --dataset imagenet \
                    --data /path/to/imagenet/ \
                    --model mobilenetv2_w1 \
                    --limit bops \
                    --limit_p 0.6 \
                    --resume models/mobilenetv2_uniform8/checkpoint.pth.tar
```

To get the result for `ResNet50`, set `limit` to `bops` and `limit_p` to `0.4`. Accuracy is reduced by 1.13%, BOPs is improved by 39.47% and latency is improved by 25.01%.
```
python3 run_test.py --dataset imagenet \
                    --data /path/to/imagenet/ \
                    --model resnet50 \
                    --limit bops \
                    --limit_p 0.4 \
                    --resume models/resnet50_uniform8/checkpoint.pth.tar
```


## Reference
This repository reference from the paper [ZeroQ: A Novel Zero-Shot Quantization Framework](https://arxiv.org/abs/2001.00281).