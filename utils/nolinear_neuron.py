import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import re
from torch.nn import functional as F


class Neurons(nn.Module):
    def __init__(self, in_features: int, out_features: int, neuron: str, bias: bool = True):
        super(Neurons, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.neuron = neuron
        self.number = neuron.count('x')

        for i in range(self.number):
            exec('self.weight{} = Parameter(torch.Tensor(out_features, in_features))'.format(i))

        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.number):
            exec('init.kaiming_uniform_(self.weight{}, a=math.sqrt(5))'.format(i))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight0)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def xdata(self):
        temp = self.neuron.replace(' ', '')
        temp = re.split('\+|-', temp)
        xx = []
        for s in temp:
            if '@' in s:
                out = s[s.find('@') + 1:]
                xx.append(out)
            elif 'x' in s:
                out = s
                xx.append(out)
            else:
                pass
        return xx

    def forward(self, x):
        xlist = self.xdata()
        assert self.number == len(xlist), 'weight length not equal to xdata'

        su = 0
        loc = locals()
        for i in range(self.number):
            exec('su += F.linear(eval(xlist[i]), self.weight{}, bias=None)'.format(i))
            su = loc['su']
        out = su + self.bias
        return out


if __name__ == '__main__':
    n = '2@x**5 - 3@x**3 + 3@x**2 + x + 1'
    a = torch.ones((3, 3))
    b = Neurons(3, 5, n)
    c = b(a)
    print(c.shape)
