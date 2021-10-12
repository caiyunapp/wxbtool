# -*- coding: utf-8 -*-

import numpy as np
import torch as th

from torch.utils.data import Dataset
from leibniz.nn.net.simple import SimpleCNN2d
from wxbtool.data.variables import vars3d
from tests.spec.spectest import Spec, Setting3d


class ModelTest(Spec):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = 'model_test'
        self.mlp = SimpleCNN2d(self.setting.input_span * len(self.setting.vars_in) + self.constant_size + 2, 1)

    def forward(self, **kwargs):
        batch_size = kwargs['2m_temperature'].size()[0]
        self.update_da_status(batch_size)

        _, input = self.get_inputs(**kwargs)
        cnst = self.get_augmented_constant(input)
        input = th.cat((input, cnst), dim=1)

        output = self.mlp(input)

        return {
            't2m': output.view(batch_size, 1, 32, 64)
        }


setting = Setting3d()
model = ModelTest(setting)
