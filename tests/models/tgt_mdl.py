# -*- coding: utf-8 -*-

'''
    Demo model in wxbtool package

    This model predict t850 3-days in the future
    The performance is relative weak, but it can be easily fitted into one normal gpu
    the weighted rmse is 2.41 K
'''

import numpy as np

from torch.utils.data import Dataset
from leibniz.nn.net.mlp import MLP2d
from wxbtool.specs.res5_625.t850weyn import Spec, Setting3d


setting = Setting3d()


class TestDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, item):
        inputs, targets = {}, {}
        for var in setting.vars:
            inputs.update({var: np.ones((1, 15, 32, 64))})
            targets.update({var: np.ones((1, 15, 32, 64))})
        return inputs, targets


class TgtMdl(Spec):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = 'tgt_mdl'
        self.mlp = MLP2d(1, 1)

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
        _ = self.get_augmented_constant(input)

        output = self.mlp(input)

        return {
            't850': output
        }


model = TgtMdl(setting)
