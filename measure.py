import torch
import torch.nn as nn
import time
from utils import Quant_Conv2d, Quant_Linear, QuantAct

measured_modules = [Quant_Conv2d, Quant_Linear, QuantAct, nn.ReLU, nn.ReLU6, nn.BatchNorm2d, nn.AvgPool2d, nn.MaxPool2d]


def save_inp_oup_data(model, layers, cali_data):
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layers, device=device)

    cached_inps = get_inp_out(cali_data)

    for name, inp in cached_inps.items():
        cached_inps[name] = inp.to(device)
    
    return cached_inps


class StopForwardException(Exception):
    pass


class GetLayerInpOut:
    def __init__(self, model, layers, device: torch.device, ):
        self.model = model
        self.layers = layers
        self.device = device
        self.inputs = {}
    
    def hook_fn(self, module, input, output, layer_name):
        self.inputs[layer_name] = input[0].detach()
    
    def register_hooks(self):
        for name, layer in self.layers:
            self.handles.append(layer.register_forward_hook(lambda module, input, output, name=name: self.hook_fn(module, input, output, name)))
        
    def __call__(self, model_input):
        self.model.eval()

        self.handles = []
        self.register_hooks()

        with torch.no_grad():
            try:
                _ = self.model(model_input.to(self.device))
            except StopForwardException:
                pass
        
        for handle in self.handles:
            handle.remove()

        return self.inputs


def measure_inference_time(model, quantized_model, test_loader):
    for _, (inputs, _) in enumerate(test_loader):
        cali_data = inputs

    layers = []
    for name, layer in model.named_modules():
        if type(layer) in measured_modules:
            layers.append((name, layer))

    cached_inps = save_inp_oup_data(model, layers, cali_data)
    
    measured_time = {}
    previous_activ_bitwidth = 8
    for name, layer in model.named_modules():
        if type(layer) in [Quant_Conv2d, Quant_Linear]:
            start_time = time.time()
            _ = layer(cached_inps[name])
            end_time = time.time()
            measured_time[name] = (previous_activ_bitwidth, getattr(layer, 'weight_bit'), end_time - start_time)
        elif isinstance(layer, QuantAct):
            previous_activ_bitwidth = getattr(layer, 'activation_bit')
            start_time = time.time()
            _ = layer(cached_inps[name])
            end_time = time.time()
            measured_time[name] = (previous_activ_bitwidth, 8, end_time - start_time)
        elif type(layer) in measured_modules:
            start_time = time.time()
            _ = layer(cached_inps[name])
            end_time = time.time()
            measured_time[name] = (8, 8, end_time - start_time)

    quantized_time ={}
    previous_activ_bitwidth = 8
    for name, layer in quantized_model.named_modules():
        if type(layer) in [Quant_Conv2d, Quant_Linear]:
            activation_bit, weight_bit, elapsed_time = measured_time[name]
            quantized_time[name] = (previous_activ_bitwidth, getattr(layer, 'weight_bit'), 
                                    elapsed_time / (activation_bit * weight_bit) * (previous_activ_bitwidth * getattr(layer, 'weight_bit')))
        elif isinstance(layer, QuantAct):
            previous_activ_bitwidth = getattr(layer, 'activation_bit')
            activation_bit, _,  elapsed_time = measured_time[name]
            quantized_time[name] = (previous_activ_bitwidth, 8, elapsed_time / activation_bit * previous_activ_bitwidth)
        elif type(layer) in measured_modules:
            _, _,  elapsed_time = measured_time[name]
            quantized_time[name] = (previous_activ_bitwidth, 8, elapsed_time)

    total_time = 0
    calculated_time = 0
    for name, (_, _, elapsed_time) in measured_time.items():
        total_time += elapsed_time
        calculated_time += quantized_time[name][2]

    return total_time / cali_data.shape[0], calculated_time / cali_data.shape[0]