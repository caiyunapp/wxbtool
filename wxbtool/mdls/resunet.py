# -*- coding: utf-8 -*-

'''

'''

import logging
import random

import torch as th
import torch.nn as nn

from leibniz.nn.net import resunet, hyptub
from leibniz.nn.activation import CappingRelu
from leibniz.unet.hyperbolic import HyperBottleneck
from wxbtool.dataset import WxDataset

logger = logging.getLogger()

step = 4
input_span = 3
pred_shift = 72

name = 'multi'
vars = ['geopotential', 'toa_incident_solar_radiation', '2m_temperature', 'temperature', 'total_cloud_cover']
levels = ['300', '500', '700', '850', '1000']
vars_in = ['z500', 'z1000', 'tau', 't850', 'tcc', 't2m', 'tisr']
vars_out = ['t850']
level = levels.index('850')
height = len(levels)


years_train = [
    1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
    1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
    2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
    2010, 2011, 2012, 2013
]
years_test = [2014]
years_eval = [2014, 2015, 2016, 2017, 2018]


dataset_train, dataset_test, dataset_eval = None, None, None
train_size = -1
test_size = -1
eval_size = -1


def load_dataset(mode='train'):
    global dataset_train, dataset_test, dataset_eval, train_size, test_size, eval_size

    if mode == 'train':
        dataset_train, dataset_eval = (
                WxDataset(mdls.root, mdls.resolution, years_train, vars, levels, input_span=input_span, pred_shift=pred_shift, pred_span=1, step=step),
                WxDataset(mdls.root, mdls.resolution, years_eval, vars, levels, input_span=input_span, pred_shift=pred_shift, pred_span=1, step=step)
        )
        train_size = dataset_train.size
        eval_size = dataset_eval.size
    else:
        dataset_test = WxDataset(mdls.root, mdls.resolution, years_test, vars, input_span=input_span, pred_shift=pred_shift, pred_span=1, step=step)
        test_size = dataset_test.size


mse = nn.MSELoss()


def linear(in_channels, out_channels, **kwargs):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


class MultiVarForecast(mdls.Model2d):
    def __init__(self):
        super().__init__()
        self.lng_shift = 0
        self.flip_status = 0

        tube = hyptub(1664, 832, 1664, encoder=linear, decoder=linear)

        self.resunet = resunet(input_span * (len(vars) + 2) + self.constant_size + 2, 1,
                            spatial=(32, 64+2), layers=5, ratio=-1,
                            vblks=[2, 2, 2, 2, 2], hblks=[1, 1, 1, 1, 1],
                            scales=[-1, -1, -1, -1, -1], factors=[1, 1, 1, 1, 1],
                            block=HyperBottleneck, relu=CappingRelu(), enhencer=tube,
                            final_normalized=False)


    def update_da_status(self, batch):
        if self.training:
            self.lng_shift = []
            self.flip_status = []
            for _ in range(batch):
                self.lng_shift.append(random.randint(0, 64))
                self.flip_status.append(random.randint(0, 1))

    def augment_data(self, data):
        if self.training:
            augmented = []
            b, c, w, h = data.size()
            for _ in range(b):
                slice = data[_:_+1]
                shift = self.lng_shift[_]
                flip = self.flip_status[_]
                slice = slice.roll(shift, dims=(3,))
                if flip == 1:
                    slice = th.flip(slice, dims=(2, 3))
                augmented.append(slice)
            data = th.cat(augmented, dim=0)
        return data

    def get_augmented_constant(self, input):
        constant = self.get_constant(input, input.device).repeat(input.size()[0], 1, 1, 1)
        constant = self.augment_data(constant)
        phi = self.get_phi(input.device).repeat(input.size()[0], 1, 1, 1)
        theta = self.get_theta(input.device).repeat(input.size()[0], 1, 1, 1)
        constant = th.cat((constant, phi, theta), dim=1)
        return constant

    def get_inputs(self, **pwargs):
        z500 = norm_z500(pwargs['geopotential'].view(-1, input_span, height, 32, 64)[:, :, levels.index('500')])
        z1000 = norm_z1000(pwargs['geopotential'].view(-1, input_span, height, 32, 64)[:, :, levels.index('1000')])
        tau = norm_tau(pwargs['geopotential'].view(-1, input_span, height, 32, 64)[:, :, levels.index('300')] - pwargs['geopotential'].view(-1, input_span, height, 32, 64)[:, :, levels.index('700')])
        t850 = norm_t850(pwargs['temperature'].view(-1, input_span, height, 32, 64)[:, :, levels.index('850')])
        tcc = norm_tcc(pwargs['total_cloud_cover'].view(-1, input_span, 32, 64))
        t2m = norm_t2m(pwargs['2m_temperature'].view(-1, input_span, 32, 64))
        tisr = norm_tisr(pwargs['toa_incident_solar_radiation'].view(-1, input_span, 32, 64))

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

    def get_targets(self, **pwargs):
        t850 = norm_t850(pwargs['temperature'].view(-1, 1, height, 32, 64)[:, :, levels.index('850')])
        t850 = self.augment_data(t850)
        return {'t850': t850}, t850

    def get_results(self, **pwargs):
        t850 = pwargs['t850']
        return {'t850': t850}, t850

    def forward(self, **pwargs):
        batch_size = pwargs['temperature'].size()[0]
        self.update_da_status(batch_size)

        _, input = self.get_inputs(**pwargs)
        constant = self.get_augmented_constant(input)
        input = th.cat((input, constant), dim=1)
        input = th.cat((input[:, :, :, 63:64], input, input[:, :, :, 0:1]), dim=3)

        output = self.resunet(input)

        output = output[:, :, :, 1:65]
        return {
            't850': output
        }


model = MultiVarForecast()


def lossfun(inputs, result, target):
    _, rst = model.get_results(**result)
    _, tgt = model.get_targets(**target)
    rst = model.weight * rst.view(-1, 1, 32, 64)
    tgt = model.weight * tgt.view(-1, 1, 32, 64)

    losst = mse(rst[:, 0], tgt[:, 0])

    return losst
