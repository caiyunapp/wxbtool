# -*- coding: utf-8 -*-

'''
    Demo model in wxbtool package

    This model predict t850 3-days in the future
    The performance is relative weak, but it can be easily fitted into one normal gpu
    the weighted rmse is 2.41 K
'''

import numpy as np
import torch as th

from torch.utils.data import Dataset
from leibniz.nn.net.simple import SimpleCNN2d
from wxbtool.data.variables import vars3d
from wxbtool.specs.res5_625.t850rasp import Spec, Setting3d


setting = Setting3d()


class TestDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, item):
        inputs, targets = {}, {}
        for var in setting.vars:
            if var in vars3d:
                inputs.update({var: np.ones((1, setting.input_span, setting.height, 32, 64))})
                targets.update({var: np.ones((1, 1, setting.height, 32, 64))})
            else:
                inputs.update({var: np.ones((1, setting.input_span, 32, 64))})
                targets.update({var: np.ones((1, 1, 32, 64))})
        return inputs, targets


class TgtMdl(Spec):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = 'tgt_mdl'
        self.mlp = SimpleCNN2d(self.setting.input_span * len(self.setting.vars_in) + self.constant_size + 2, 1)

    def load_dataset(self, phase, mode, **kwargs):
        self.phase = phase
        self.mode = mode

        self.dataset_train = TestDataset()
        self.dataset_eval = TestDataset()
        self.dataset_test = TestDataset()

        self.train_size = len(self.dataset_train)
        self.eval_size = len(self.dataset_eval)
        self.test_size = len(self.dataset_test)

    def forward(self, **kwargs):
        batch_size = kwargs['temperature'].size()[0]
        self.update_da_status(batch_size)

        _, input = self.get_inputs(**kwargs)
        cnst = self.get_augmented_constant(input)
        input = th.cat((input, cnst), dim=1)

        output = self.mlp(input)

        return {
            't850': output.view(batch_size, 1, 32, 64)
        }


model = TgtMdl(setting)
