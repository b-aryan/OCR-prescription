import torch
from torchsummary import summary
from crnn import CRNN
import torch.nn as nn
from collections import OrderedDict


def print_model_summary(model, input_size):
    def forward_hook(module, input, output):
        class_name = str(module.__class__.__name__)
        module_index = len(summary)

        m_key = f"{class_name}-{module_index + 1}"
        summary[m_key] = OrderedDict()
        summary[m_key]["input_shape"] = str(input[0].shape)
        summary[m_key]["output_shape"] = str(output.shape)
        summary[m_key]["num_params"] = sum(p.numel() for p in module.parameters())

        print(f"Layer: {m_key}, Input Shape: {input[0].shape}, Output Shape: {output.shape}")

        print(f"Output Shape Components: {[dim for dim in output.shape]}")

        if isinstance(module, nn.Sequential):
            for idx, submodule in enumerate(module):
                input_shape = str(input[idx + 1].shape) if idx + 1 < len(input) else None
                output_shape = str(output[idx].shape) if idx < len(output) else None
                print(f"  - Submodule {idx + 1}: Input Shape - {input_shape}, Output Shape - {output_shape}")

    device = next(model.parameters()).device
    summary = OrderedDict()

    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(forward_hook)
        hooks.append(hook)

    model.to(device)
    model(torch.zeros(1, *input_size).to(device))

    for hook in hooks:
        hook.remove()

    print("\n" + "=" * 140)
    print(f"{'Layer': <25}{'Input Shape': <40}{'Output Shape': <40}{'Param #': <20}")
    print("=" * 140)

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        input_shape, output_shape, num_params = summary[layer]["input_shape"], summary[layer]["output_shape"], \
            summary[layer]["num_params"]
        try:

            total_output += abs(torch.prod(torch.tensor([int(dim) for dim in output_shape])).item())
        except ValueError as e:

            # print(f"Error in layer {layer}: {e}")
            pass
        total_params += num_params
        if "conv" in layer.lower() or "linear" in layer.lower():
            trainable_params += num_params
        print(f"{layer: <25}{input_shape: <40}{output_shape: <40}{num_params: <20}")

    print("=" * 140)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")
    print(f"Non-trainable params: {total_params - trainable_params}")
    print(f"Total output: {total_output}")
    print("=" * 140 + "\n")


imgH, nc, nclass, nh = 32, 1, 10, 256
crnn_model = CRNN(imgH, nc, nclass, nh)

print_model_summary(crnn_model, (nc, imgH, 128))