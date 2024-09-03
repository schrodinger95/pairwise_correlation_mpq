import argparse
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model

import utils
from distill_data import *
from nsga import nsga
from bit_config import bit_config
from measure import measure_inference_time
from calculate import calculate_bops_mobilenet, calculate_size_mobilenet, calculate_bops_resnet50, calculate_size_resnet50

caculate_dict = {'mobilenetv2_w1': {'bops': calculate_bops_mobilenet, 'size': calculate_size_mobilenet},
                 'resnet50': {'bops': calculate_bops_resnet50, 'size': calculate_size_resnet50}}


def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=['imagenet', 'cifar10'],
                        help='type of dataset')
    parser.add_argument('--data', metavar='DIR',
                        default='/data/imagenet/',
                        help='path to dataset')
    parser.add_argument('--model',
                        type=str,
                        default='mobilenetv2_w1',
                        choices=['resnet50', 'mobilenetv2_w1'],
                        help='model to be quantized')
    parser.add_argument('--limit',
                        type=str,
                        default='bops',
                        choices=['bops', 'size'],
                        help='limit BOPs or model size')
    parser.add_argument('--limit_p',
                        type=float,
                        default=0.8,
                        help='limit percentile for BOPs or model size')
    parser.add_argument('--p',
                        type=float,
                        default=0.75,
                        help='percentile of outliers')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='batch size of test data')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    return args


def solve(arch, limit, limit_p, q):
    if arch == 'mobilenetv2_w1':
        df = pd.read_csv('database/MobilenetV2_SingleLayer.csv')
        pairwise_df = pd.read_csv('database/MobilenetV2_PairwiseLayers.csv', index_col=0)
        multiple_df = pd.read_csv('database/MobilenetV2_MultipleLayers.csv')
    else:
        df = pd.read_csv('database/Resnet50_SingleLayer.csv')
        pairwise_df = pd.read_csv('database/Resnet50_PairwiseLayers.csv', index_col=0)
        multiple_df = pd.read_csv('database/Resnet50_MultipleLayers.csv')

    delta_accs = df['DeltaAcc'].values
    quantile = np.quantile(delta_accs, q=q)
    layers = df['Layer'].values
    df['Influence_fillna'] = df['Influence'].fillna(100)

    pairwise_df = pairwise_df.sort_index()
    pairwise_df = pairwise_df.dropna()
    pairwise_df['XY-X-Y'] = pairwise_df['Delta_acc'] - pairwise_df['Delta_acc1'] - pairwise_df['Delta_acc2']
    df_filtered = pairwise_df[(pairwise_df['Delta_acc1'] < quantile) & (pairwise_df['Delta_acc2'] < quantile)]

    multiple_df = multiple_df.dropna()
    multiple_df['XY-X-Y'] = multiple_df['Delta Acc'] - multiple_df['F1']

    layer_num_dict = {'Layer_num':[0, 1, 2, ], 'Avg_corr': [0, 0, df_filtered['XY-X-Y'].values.mean()]}
    for layer_num in sorted(np.unique(multiple_df['Layer_num'].values)):
        avg_corr = multiple_df.loc[multiple_df['Layer_num'] == layer_num, 'XY-X-Y'].mean()
        layer_num_dict['Layer_num'].append(layer_num)
        layer_num_dict['Avg_corr'].append(avg_corr)
    max_layer = max(layer_num_dict['Layer_num'])
    for layer_num in range(2, max_layer):
        if layer_num not in layer_num_dict['Layer_num']:
            avg_corr = (layer_num_dict['Avg_corr'][layer_num_dict['Layer_num'].index(layer_num - 1)] + layer_num_dict['Avg_corr'][layer_num_dict['Layer_num'].index(layer_num + 1)]) / 2
            layer_num_dict['Layer_num'].append(layer_num)
            layer_num_dict['Avg_corr'].append(avg_corr)
    for layer_num in range(max_layer + 1, len(layers) + 1):
        layer_num_dict['Layer_num'].append(layer_num)
        layer_num_dict['Avg_corr'].append(layer_num_dict['Avg_corr'][layer_num_dict['Layer_num'].index(max_layer)] )
    layer_num_df = pd.DataFrame(layer_num_dict)

    output_bit_config = nsga(arch, limit, limit_p, df, layer_num_df)
    return output_bit_config


def preprocess_config(arch, bit_config):
    ba = []
    bw = []
    if arch == "mobilenetv2_w1":
        for name, bitwidth in bit_config.items():
            if "features.stage" in name and "quant_act_int32" not in name:
                if 'quant_act' in name:
                    ba.append(bitwidth)
                else:
                    bw.append(bitwidth)
    elif arch == "resnet50":
        for name, bitwidth in bit_config.items():
            if "stage" in name and "quant_act_int32" not in name and "identity" not in name:
                if 'quant_act' in name:
                    ba.append(bitwidth)
                else:
                    bw.append(bitwidth)
    return ba, bw

def load_state_dict(resume, arch, model):
    checkpoint = torch.load(resume)['state_dict']
    modified_dict = {}
    for key, value in checkpoint.items():
        if 'weight_integer' in key: continue
        if 'bias_integer' in key: continue
        if 'quant_act' in key: continue
        if 'scaling_factor' in key: continue
        if 'num_batches_tracked' in key: continue
        if 'min' in key or 'max' in key: continue

        modified_key = key.replace("module.", "")
        if 'feature' not in key and 'output' not in key:
            modified_key = 'features.' + modified_key

        if arch == 'mobilenetv2_w1':
            if 'output.conv.weight' in key:
                continue

        elif arch == 'resnet50':
            modified_key = modified_key.replace("quant_convbn", "conv")
            modified_key = modified_key.replace("quant_identity_convbn", "identity_conv")
            modified_key = modified_key.replace("quant_init_convbn", "init_block.conv")
            modified_key = modified_key.replace("quant_output", "output")
            if 'stage' in key and 'identity' not in key:
                modified_key = modified_key[:21] + '.body' + modified_key[21:]
        
        modified_dict[modified_key] = value
    model.load_state_dict(modified_dict)


def load_bit_config(arch, model, bit_config):
    for name, m in model.named_modules():
        setattr(m, 'name', name)

        if type(m) in [nn.Conv2d, nn.Linear, nn.ReLU, nn.ReLU6]:
            if arch == 'mobilenetv2_w1':
                if 'init_block' in name:
                    bitwidth = bit_config['init_block']
                elif 'final_block' in name:
                    bitwidth = bit_config['features.final_block']
                elif 'output' in name:
                    bitwidth = bit_config[name]
                else:
                    if 'activ' in name:
                        bitwidth = bit_config[name[:name.index('.conv')] + '.quant_act' + name[name.index('.activ') - 1]]
                    else:
                        bitwidth = bit_config[name[:len(name) - 1 - name[::-1].index('.')]]
            if arch == 'resnet50':
                if 'init_block' in name:
                    bitwidth = bit_config['quant_init_convbn']
                elif 'stage' in name:
                    if 'identity' in name:
                        bitwidth = bit_config[name[name.index('stage'):name.index('.identity')] + '.quant_identity_convbn']
                    elif '.activ' not in name:
                        bitwidth = bit_config[name[name.index('stage'):name.index('.body')] + '.quant_convbn' + name[name.index('.conv') + 5]]
                    elif 'body' in name:
                        bitwidth = bit_config[name[name.index('stage'):name.index('.body')] + '.quant_act' + name[name.index('.conv') + 5]]
                    else:
                        bitwidth = 8
                else:
                    bitwidth = bit_config['quant_output']
            setattr(m, 'bitwidth', bitwidth)


if __name__ == '__main__':
    args = arg_parse()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Load validation data
    test_loader = utils.getTestData(args.dataset,
                                    batch_size=args.test_batch_size,
                                    path=args.data,
                                    for_inception=args.model.startswith('inception'))
    print('****** Test Data Loaded ******')

    # Load pretrained model
    model = ptcv_get_model(args.model, pretrained=False)
    if args.resume:
        load_state_dict(args.resume, args.model, model)

    # Load 8-bit configuration
    load_bit_config(args.model, model, bit_config[args.model])
    print('****** Full-Precision Model Loaded ******')

    # Quantize single-precision model to 8-bit model
    uniform_model = utils.quantize_model(model)

    # Freeze BatchNorm statistics
    uniform_model.eval()
    uniform_model = uniform_model.cuda()

    # Freeze activation range during test
    utils.freeze_model(uniform_model)
    uniform_model = nn.DataParallel(uniform_model).cuda()
    print('****** 8-bit Model Loaded ******')

    # Solve the optimal mixed-precision scheme and load the bit-width configuration
    output_bit_config = list(solve(args.model, args.limit, args.limit_p, args.p).values())[0]
    load_bit_config(args.model, model, output_bit_config)
    
    # Generate distilled data
    dataloader = getDistilData(
        model.cuda(),
        args.dataset,
        batch_size=args.batch_size,
        for_inception=args.model.startswith('inception'))
    print('****** Distilled Data Loaded ******')

    # Quantize single-precision model to 8-bit model
    quantized_model = utils.quantize_model(model)

    # Freeze BatchNorm statistics
    quantized_model.eval()
    quantized_model = quantized_model.cuda()

    # Update activation range according to distilled data
    utils.update(quantized_model, dataloader)
    print('****** Zero Shot Quantization Finished ******')

    # Freeze activation range during test
    utils.freeze_model(quantized_model)
    quantized_model = nn.DataParallel(quantized_model).cuda()
    print('****** Mixed-Precision Model Loaded ******')
    
    print('Testing ...')
    # Test the final quantized model
    result = {'uniform': {}, 'mixed': {}}

    ba, bw = preprocess_config(args.model, bit_config[args.model])
    result['uniform']['bops'] = caculate_dict[args.model]['bops'](ba, bw)
    ba, bw = preprocess_config(args.model, output_bit_config)
    result['mixed']['bops'] = caculate_dict[args.model]['bops'](ba, bw)

    uniform_time, quantized_time = measure_inference_time(uniform_model, quantized_model, test_loader)
    result['uniform']['latency'] = uniform_time
    result['mixed']['latency'] = quantized_time

    result['uniform']['acc'] = 100 * utils.test(uniform_model, test_loader)
    result['mixed']['acc'] = 100 * utils.test(quantized_model, test_loader)

    print(f"8-bit Model:\n\tAcc: {result['uniform']['acc']:.2f}, BOPs: {result['uniform']['bops']:.2f}, latency: {result['uniform']['latency']:.2e}")
    print(f"Mixed-Precision Model:\n\tAcc: {result['mixed']['acc']:.2f}, BOPs: {result['mixed']['bops']:.2f}, latency: {result['mixed']['latency']:.2e}")
    print(f"Accuracy degrade by {result['uniform']['acc'] - result['mixed']['acc']:.2f}%")
    print(f"BOPs improved by {(result['uniform']['bops'] - result['mixed']['bops']) / result['uniform']['bops'] * 100:.2f}%")
    print(f"latency improved by {(result['uniform']['latency'] - result['mixed']['latency']) / result['uniform']['latency'] * 100:.2f}%")
    print('****** Test Finished ******')
