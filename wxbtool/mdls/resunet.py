# -*- coding: utf-8 -*-

'''
 Demo model in wxbtool package
'''

import logging

import torch as th
import torch.nn as nn

from leibniz.nn.net import resunet
from leibniz.nn.activation import CappingRelu
from leibniz.unet.senet import SEBottleneck

from wxbtool.norms.meanstd import *
from wxbtool.nn.setting import Setting
from wxbtool.nn.model import Base2d

logger = logging.getLogger()


mse = nn.MSELoss()


class ModelSetting(Setting):
    def __init__(self):
        super().__init__()
        self.resolution = '5.625deg'    # The spatial resolution of the model

        self.name = 'resunet'           # The name of the model

        self.step = 4                   # How many hours of a hourly step which all features in organized temporally
        self.input_span = 3             # How many hourly steps for an input
        self.pred_span = 1              # How many hourly steps for a prediction
        self.pred_shift = 72            # How many hours between the end of the input span and the beginning of prediction span

        self.levels = ['300', '500', '700', '850', '1000'] # Which vertical levels to choose
        self.height = len(self.levels)                     # How many vertical levels to choose

        # The name of variables to choose, for both input features and output
        self.vars = ['geopotential', 'toa_incident_solar_radiation', '2m_temperature', 'temperature', 'total_cloud_cover']

        # The code of variables in input features
        self.vars_in = ['z500', 'z1000', 'tau', 't850', 'tcc', 't2m', 'tisr']
        # The code of variables in output
        self.vars_out = ['t850']

        # temporal scopes for train
        self.years_train = [
            1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
            1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
            2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
            2010, 2011, 2012, 2013, 2014,
        ]
        # temporal scopes for evaluation
        self.years_eval = [2015, 2016]
        # temporal scopes for test
        self.years_test = [2017, 2018]


class Enhencer(nn.Module):
    def __init__(self, channels):
        super(Enhencer, self).__init__()
        hidden = channels * 2
        self.fci = nn.Linear(channels, hidden)
        self.fco = nn.Linear(hidden, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.fci(x.view(b, c * h * w))
        out = self.relu(out)
        out = self.fco(out).view(b, c, h, w)
        return out


class ResUNetModel(Base2d):
    def __init__(self, setting):
        super().__init__(setting)
        enhencer = Enhencer(3328)
        self.resunet = resunet(setting.input_span * (len(setting.vars) + 2) + self.constant_size + 2, 1,
                            spatial=(32, 64+2), layers=5, ratio=-1,
                            vblks=[2, 2, 2, 2, 2], hblks=[1, 1, 1, 1, 1],
                            scales=[-1, -1, -1, -1, -1], factors=[1, 1, 1, 1, 1],
                            block=SEBottleneck, relu=CappingRelu(), enhencer=enhencer,
                            final_normalized=False)

    def get_inputs(self, **kwargs):
        z500 = norm_z500(kwargs['geopotential'].view(-1, self.setting.input_span, self.setting.height, 32, 64)[:, :, self.setting.levels.index('500')])
        z1000 = norm_z1000(kwargs['geopotential'].view(-1, self.setting.input_span, self.setting.height, 32, 64)[:, :, self.setting.levels.index('1000')])
        tau = norm_tau(kwargs['geopotential'].view(-1, self.setting.input_span, self.setting.height, 32, 64)[:, :, self.setting.levels.index('300')] - kwargs['geopotential'].view(-1, self.setting.input_span, self.setting.height, 32, 64)[:, :, self.setting.levels.index('700')])
        t850 = norm_t850(kwargs['temperature'].view(-1, self.setting.input_span, self.setting.height, 32, 64)[:, :, self.setting.levels.index('850')])
        tcc = norm_tcc(kwargs['total_cloud_cover'].view(-1, self.setting.input_span, 32, 64))
        t2m = norm_t2m(kwargs['2m_temperature'].view(-1, self.setting.input_span, 32, 64))
        tisr = norm_tisr(kwargs['toa_incident_solar_radiation'].view(-1, self.setting.input_span, 32, 64))

        z500 = self.augment_data(z500)
        z1000 = self.augment_data(z1000)
        tau = self.augment_data(tau)
        t850 = self.augment_data(t850)
        tcc = self.augment_data(tcc)
        t2m = self.augment_data(t2m)
        tisr = self.augment_data(tisr)

        return {
            'z500': z500,
            'z1000': z1000,
            'tau': tau,
            't850': t850,
            'tcc': tcc,
            't2m': t2m,
            'tisr': tisr,
        }, th.cat((
            z500, z1000,
            tau, t850, tcc,
            t2m, tisr,
        ), dim=1)

    def get_targets(self, **kwargs):
        t850 = kwargs['temperature'].view(-1, 1, self.setting.height, 32, 64)[:, :, self.setting.levels.index('850')]
        t850 = self.augment_data(t850)
        return {'t850': t850}, t850

    def get_results(self, **kwargs):
        t850 = denorm_t850(kwargs['t850'])
        return {'t850': t850}, t850

    def forward(self, **kwargs):
        batch_size = kwargs['temperature'].size()[0]
        self.update_da_status(batch_size)

        _, input = self.get_inputs(**kwargs)
        constant = self.get_augmented_constant(input)
        input = th.cat((input, constant), dim=1)
        input = th.cat((input[:, :, :, 63:64], input, input[:, :, :, 0:1]), dim=3)

        output = self.resunet(input)

        output = output[:, :, :, 1:65]
        return {
            't850': output
        }

    def lossfun(self, inputs, result, target):
        _, rst = self.get_results(**result)
        _, tgt = self.get_targets(**target)
        rst = self.weight * rst.view(-1, 1, 32, 64)
        tgt = self.weight * tgt.view(-1, 1, 32, 64)

        losst = mse(rst[:, 0], tgt[:, 0])

        return losst


setting = ModelSetting()
model = ResUNetModel(setting)
