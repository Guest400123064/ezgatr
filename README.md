<html>
    <h1 align="center">
      <img src="https://raw.githubusercontent.com/Guest400123064/ezgatr/refs/heads/main/docs/images/ezgatr_logo.png" width="256"/>
    </h1>
    <h3 align="center">
      Geometric Algebra Transformer Made Easy
    </h3>
</html>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-%203.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/downloads/release/python-3100/)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)
[![Website](https://img.shields.io/website?label=documentation&up_message=online&url=https%3A%2F%2Fguest400123064.github.io/ezgatr)](https://guest400123064.github.io/ezgatr/ezgatr.html)

## What is EzGATr?
**EzGATr** (Easy Geometric Algebra Transformer) intends to be a simple-to-use and lightweight Python library for building 3D [Geometric Algebra Transformers (GATr)](https://arxiv.org/abs/2305.18415). It is a collection of operators, modules, utilities, etc. build on top of [PyTorch](https://pytorch.org/). In addition, EzGATr also seeks to bridge the gap between the mathematical formulations and corresponding implementations through extensive documentation and explanations to facilitate learning and potential future optimizations.

## Installation
EzGATr is currently in development and not yet available on PyPI. To install it, you can clone the repository and install it using `pip`. Use the `-e` flag to install it in editable mode for quick changes.

```bash
$ git clone https://github.com/Guest400123064/ezgatr.git
$ cd ezgatr
$ pip install -e .
```

## Usage
The quick start example shown below demonstrates how to use EzGATr to build a equivariant network with one input/output channel (e.g., a point cloud) and four hidden channels. The mock input data contains a batch of eight samples, each with 256 3D objects embedded as multi-vectors.

```python
import torch
import torch.nn as nn

from ezgatr.nn import EquiLinear
from ezgatr.nn.functional import scaler_gated_gelu


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = EquiLinear(1, 4)
        self.fc2 = EquiLinear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = scaler_gated_gelu(x)
        x = self.fc2(x)
        return x


net = SimpleNet()
in_ = torch.randn(8, 256, 1, 16)
out = net(in_)
```

One can refer to [this example](https://github.com/Guest400123064/ezgatr/blob/main/src/ezgatr/nets/mv_only_gatr.py) for how to build a full-fledged GATr model with EzGATr, involving equivariant geometric attention and geometric MLP.

## API References
The complete API references for EzGATr can be found [here](https://guest400123064.github.io/ezgatr/ezgatr.html). TL;DR, the package is organized as follows:

* `ezgatr.nn`: Contains the core modules and layers for building GATr models. It is organized similarly to PyTorch's `torch.nn` package, where the functional submodule contains lower-level operators and transformations.
* `ezgatr.interfaces`: Contains the utility functions that help encode and decode 3D objects to and from multi-vectors.
* `ezgatr.nets`: Contains off-of-the-shelf networks built with EzGATr building blocks. It can also be used as references for building custom networks with EzGATr.

## License
EzGATr is distributed under the terms of the [MIT](https://opensource.org/licenses/MIT) license.
