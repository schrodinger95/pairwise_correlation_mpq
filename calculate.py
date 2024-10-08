import numpy as np


def MAC_mobilenet(c_i, k, hw, c_o):
    if k == 3:
        flot = c_i * 0.5 * (2 * k * k - 1) * hw
    else:
        flot = 0.5 * (2 * c_i * k * k - 1) * hw * c_o
    return flot


def calculate_bops_mobilenet(ba, bw):
    c_i = [3,32,32,32,16,96,96,24,144,144,24,144,144,32,192,192,32,192,192,32,192,192,64,384,384,64,384,384,64,384,384,64,384,384,96,576,576,96,576,576,96,576,576,160,960,960,160,960,960,160,960,960,320,1280]
    c_o = [32,32,32,16,96,96,24,144,144,24,144,144,32,192,192,32,192,192,32,192,192,64,384,384,64,384,384,64,384,384,64,384,384,96,576,576,96,576,576,96,576,576,160,960,960,160,960,960,160,960,960,320,1280,1000]
    hw = [50176,12544,12544,12544,12544,12544,12544,12544,12544,12544,3136,3136,3136,3136,3136,3136,3136,3136,3136,784,784,784,784,784,784,784,784,784,196,196,196,196,196,196,196,196,196,196,196,196,196,196,196,196,196,196,196,196,196,49,49,49,1,1]
    k_size = [3,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,1]

    sum = 0
    for k in range(54):
        if k in [0, 52, 53]:
            bops = 8 * 8 * MAC_mobilenet(c_i[k], k_size[k], hw[k] ,c_o[k])
        else:
            bops = ba[k - 1] * bw[k - 1] * MAC_mobilenet(c_i[k], k_size[k], hw[k] ,c_o[k])
        sum += bops
    sum /= 1e9
    return sum


def calculate_size_mobilenet(ba, bw):
    in_channel = [3, 32, 32, 32, 16, 96, 96, 24, 144, 144, 24, 144, 144, 32, 192, 192, 32, 192, 192, 32, 192, 192, 64, 384, 384, 64, 384, 384, 64, 384, 384, 64, 384, 384, 96, 576, 576, 96, 576, 576, 96, 576, 576, 160, 960, 960, 160, 960, 960, 160, 960, 960, 320, 1280]
    out_channel = [32, 32, 32, 16, 96, 96, 24, 144, 144, 24, 144, 144, 32, 192, 192, 32, 192, 192, 32, 192, 192, 64, 384, 384, 64, 384, 384, 64, 384, 384, 64, 384, 384, 96, 576, 576, 96, 576, 576, 96, 576, 576, 160, 960, 960, 160, 960, 960, 160, 960, 960, 320, 1280, 1000]
    kernel = [3, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1]
    group = [1, 1, 32, 1, 1, 96, 1, 1, 144, 1, 1, 144, 1, 1, 192, 1, 1, 192, 1, 1, 192, 1, 1, 384, 1, 1, 384, 1, 1, 384, 1, 1, 384, 1, 1, 576, 1, 1, 576, 1, 1, 576, 1, 1, 960, 1, 1, 960, 1, 1, 960, 1, 1, 1]

    sum = 0
    for i in range(54):
        if i in [0, 52, 53]:
            size = kernel[i] ** 2 * (in_channel[i] / group[i]) * out_channel[i]
        else:
            size = kernel[i] ** 2 * (in_channel[i] / group[i]) * out_channel[i] / 8 * bw[i - 1]
        sum += size
    sum /= 1024 * 1024
    return sum


def MAC_resnet50(c_i, k, hw, c_o):
    flot = 0.5 * (2 * c_i * k * k - 1) * hw * c_o
    return flot


def calculate_bops_resnet50(ba, bw):
    c_i = [3, 64, 64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 256, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 512, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 1024, 2048, 512, 512, 2048, 512, 512]
    c_o = [64, 64, 64, 256, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 2048, 512, 512, 2048, 512, 512, 2048]
    hw = [12544, 3136, 3136, 3136, 3136, 3136, 3136, 3136, 3136, 3136, 3136, 784, 784, 784, 784, 784, 784, 784, 784, 784, 784, 784, 784, 784, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49]
    k_size = [7, 1, 3, 1, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 1, 3, 1]

    sum = 0
    index = 0
    for k in range(53):
        if k in [0]:
            bops = 8 * 8 * MAC_resnet50(c_i[k], k_size[k], hw[k] ,c_o[k])
        elif k in [4, 14, 27, 46]:
            bops = 8 * 8 * MAC_resnet50(c_i[k], k_size[k], hw[k] ,c_o[k])
        else:
            bops = ba[index] * bw[index] * MAC_resnet50(c_i[k], k_size[k], hw[k] ,c_o[k])
            index += 1
        sum += bops
    sum /= 1e9
    return sum


def calculate_size_resnet50(ba, bw):
    in_channel = [3, 64, 64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 256, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 512, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 1024, 2048, 512, 512, 2048, 512, 512]
    out_channel = [64, 64, 64, 256, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 2048, 512, 512, 2048, 512, 512, 2048]
    kernel = [7, 1, 3, 1, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 1, 3, 1]

    sum = 0
    index = 0
    for i in range(53):
        if i in [0]:
            size = kernel[i] ** 2 * in_channel[i] * out_channel[i]
        elif i in [4, 14, 27, 46]:
            size = kernel[i] ** 2 * in_channel[i] * out_channel[i]
        else:
            size = kernel[i] ** 2 * in_channel[i] * out_channel[i] / 8 * bw[index]
            index += 1
        sum += size
    sum += out_channel[-1] * 1000
    sum /= 1024 * 1024
    return sum