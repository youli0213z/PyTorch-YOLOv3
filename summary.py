import torch
from thop import clever_format, profile
from torchsummary import summary
from torch import nn
from pytorchyolo.models import load_model


device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = 'config/yolov3-custom.cfg'
model = load_model(model).to(device)
summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

input_shape     = [608, 608]
dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)

flops           = flops
flops, params   = clever_format([flops, params], "%.3f")
print('Total GFLOPS: %s' % (flops))
print('Total params: %s' % (params))

