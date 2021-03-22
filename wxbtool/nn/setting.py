import wxbtool.config as config


class Setting:
    def __init__(self):
        self.root = config.root         # The root path of WeatherBench Dataset, inject from config
        self.resolution = '5.625deg'    # The spatial resolution of the model

        self.name = 'test'              # The name of the model

        self.step = 4
        self.input_span = 3
        self.pred_span = 1
        self.pred_shift = 72

        self.levels = ['300', '500', '700', '850', '1000']
        self.height = len(self.levels)

        self.vars = ['geopotential', 'toa_incident_solar_radiation', '2m_temperature', 'temperature', 'total_cloud_cover']
        self.vars_in = ['z500', 'z1000', 'tau', 't850', 'tcc', 't2m', 'tisr']
        self.vars_out = ['t850', 'z500']

        self.years_train = [
            1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
            1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
            2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
            2010, 2011, 2012, 2013, 2014,
        ]
        self.years_test = [2015]
        self.years_eval = [2016, 2017, 2018]
