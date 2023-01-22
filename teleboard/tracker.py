import re
from collections import defaultdict

import neptune.new as neptune
import numpy as np
import torch
import torch.nn.functional as F

from teleboard.distributions import QuantileDistribution


def tensor_hash(tensor):
    vector = tensor.flatten()
    x = vector.sum()
    y = vector[0::2].sum() - vector[1::2].sum()
    return x, y


class TrackerBase:
    def __init__(self, k):
        self.k = k
        self.outputs = {}
        self.last = {}

    def _activatioh_hook(self, name):
        def hook(_model, _input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.outputs[name] = output.detach()

        return hook

    def set_hooks(self, model):
        for name, module in model.named_modules():
            module.register_forward_hook(self._activatioh_hook(name))

    def statistics(self, name, param, with_xy=False):
        assert isinstance(param, torch.Tensor)
        value = param.detach().clone()
        eps = 1e-8
        previous = self.last.get(name, value)
        self.tensor(f"{name}:log", (value.abs() + eps).log())
        self.scalar(
            f"{name}:cosine_sim",
            F.cosine_similarity(value.flatten(), previous.flatten(), dim=0).item(),
        )
        if with_xy:
            x, y = tensor_hash(value)
            self.scalar(f"{name}:x", x)
            self.scalar(f"{name}:y", y)
        self.last[name] = value.detach()

    def tensor(self, name, value):
        assert isinstance(value, torch.Tensor)
        qd = QuantileDistribution(value.detach().cpu().numpy().ravel(), self.k)
        for i in range(self.k + 1):
            self.scalar(f"{name}%{i/self.k:.3f}", qd.q[i])

    def model(self, model):
        for i, (name, param) in enumerate(model.named_parameters()):
            self.statistics(
                f"{i}_{name}:dweight",
                param - self.last.get(f"{i}_{name}:weight", param),
            )
            self.statistics(f"{i}_{name}:weight", param, with_xy=True)
            self.statistics(f"{i}_{name}:gradient", param.grad if param.grad is not None else torch.zeros_like(param))

        for i, (name, _module) in enumerate(model.named_modules()):
            self.tensor(f"{i}_{name}:output", self.outputs.get(name, torch.zeros(1)))

    def scalar(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        self._scalar(name, value)

    def _scalar(self, name, value):
        pass

    def dump(self):
        pass


class NeptuneTracker(TrackerBase):
    def __init__(self, k, project, api_token):
        super().__init__(k)
        self.run = neptune.init_run(project, api_token)

    def _scalar(self, name, value):
        self.run[name].log(value)


class FileTracker(TrackerBase):
    def __init__(self, k, filename):
        super().__init__(k)
        self.series = defaultdict(list)
        self.filename = filename

    def _scalar(self, name, value):
        self.series[name].append(value)

    def dump(self):
        compressed = {
            name: np.array(value, dtype=np.float32)
            for name, value in self.series.items()
        }
        torch.save(compressed, self.filename)


class ConsoleTracker(TrackerBase):
    def __init__(self, k, regex):
        super().__init__(k)
        self.series = defaultdict(list)
        self.regex = regex

    def _scalar(self, name, value):
        self.series[name].append(value)
        if re.fullmatch(self.regex, name):
            s = self.series[name]
            print(f"{name} = {float(value):.4f}, mean = {float(sum(s) / len(s)):.4f}")
