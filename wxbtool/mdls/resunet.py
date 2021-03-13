# -*- coding: utf-8 -*-

'''
 Demo model in wxbtool package
'''

import logging

import torch as th
import torch.nn as nn

from leibniz.nn.net import resunet, hyptub
from leibniz.nn.activation import CappingRelu
from leibniz.unet.hyperbolic import HyperBottleneck

from wxbtool.norms.meanstd import *
from wxbtool.model import Base2d

logger = logging.getLogger()


mse = nn.MSELoss()


def linear(in_channels, out_channels, **kwargs):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


class Setting:
    def __init__(self, root='weatherbench/5.625deg/', resolution='5.625deg', name='resunet'):
        self.root = root
        self.resolution = resolution

        self.name = name

        self.step = 4
        self.input_span = 3
        self.pred_span = 1
        self.pred_shift = 72

        self.levels = ['300', '500', '700', '850', '1000']
        self.height = len(self.levels)

        self.vars = ['geopotential', 'toa_incident_solar_radiation', '2m_temperature', 'temperature', 'total_cloud_cover']
        self.params_in = ['z500', 'z1000', 'tau', 't850', 'tcc', 't2m', 'tisr']
        self.params_out = ['t850', 'z500']

        self.years_train = [
            1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
            1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
            2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
            2010, 2011, 2012, 2013, 2014,
        ]
        self.years_test = [2015]
        self.years_eval = [2016, 2017, 2018]


class MultiVarForecast(Base2d):
    def __init__(self, setting):
        super().__init__(setting)

        tube = hyptub(1664, 832, 1664, encoder=linear, decoder=linear)
        self.resunet = resunet(setting.input_span * (len(setting.vars) + 2) + self.constant_size + 2, 1,
                            spatial=(32, 64+2), layers=5, ratio=-1,
                            vblks=[2, 2, 2, 2, 2], hblks=[1, 1, 1, 1, 1],
                            scales=[-1, -1, -1, -1, -1], factors=[1, 1, 1, 1, 1],
                            block=HyperBottleneck, relu=CappingRelu(), enhencer=tube,
                            final_normalized=False)

    def get_inputs(self, **pwargs):
        z500 = norm_z500(pwargs['geopotential'].view(-1, self.setting.input_span, self.setting.height, 32, 64)[:, :, self.setting.levels.index('500')])
        z1000 = norm_z1000(pwargs['geopotential'].view(-1, self.setting.input_span, self.setting.height, 32, 64)[:, :, self.setting.levels.index('1000')])
        tau = norm_tau(pwargs['geopotential'].view(-1, self.setting.input_span, self.setting.height, 32, 64)[:, :, self.setting.levels.index('300')] - pwargs['geopotential'].view(-1, self.setting.input_span, self.setting.height, 32, 64)[:, :, self.setting.levels.index('700')])
        t850 = norm_t850(pwargs['temperature'].view(-1, self.setting.input_span, self.setting.height, 32, 64)[:, :, self.setting.levels.index('850')])
        tcc = norm_tcc(pwargs['total_cloud_cover'].view(-1, self.setting.input_span, 32, 64))
        t2m = norm_t2m(pwargs['2m_temperature'].view(-1, self.setting.input_span, 32, 64))
        tisr = norm_tisr(pwargs['toa_incident_solar_radiation'].view(-1, self.setting.input_span, 32, 64))

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
        t850 = norm_t850(kwargs['temperature'].view(-1, 1, self.setting.height, 32, 64)[:, :, self.setting.levels.index('850')])
        t850 = self.augment_data(t850)
        return {'t850': t850}, t850

    def get_results(self, **kwargs):
        t850 = kwargs['t850']
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
