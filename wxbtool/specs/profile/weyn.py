# -*- coding: utf-8 -*-

'''
 A modeling profile: a selection of features in different levels

 This profile follows basic settings and discussions in

   Improving Data‐Driven Global Weather Prediction Using Deep Convolutional Neural Networks on a Cubed Sphere
   by Jonathan A. Weyn, Dale R. Durran, Rich Caruana
   https://doi.org/10.1029/2020MS002109

'''

from wxbtool.nn.setting import Setting


class ProfileWeyn(Setting):
    def __init__(self):
        super().__init__()

        self.levels = ['300', '500', '700', '850', '1000'] # Which vertical levels to choose
        self.height = len(self.levels)                     # How many vertical levels to choose

        # The name of variables to choose, for both input features and output
        self.vars = ['geopotential', 'toa_incident_solar_radiation', '2m_temperature', 'temperature']

        # The code of variables in input features
        self.vars_in = ['z500', 'z1000', 'tau', 't850', 't2m', 'tisr']
        # The code of variables in output
        self.vars_out = ['t850']
