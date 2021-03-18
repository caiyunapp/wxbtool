# -*- coding: utf-8 -*-

import argparse
import importlib
import logging
import resource
import arrow
import sys
import os
import flask

import msgpack
import msgpack_numpy as m
m.patch()

import numpy as np

from pathlib import Path
from flask import Flask


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--ip", type=str, default='127.0.0.1', help="the ip of the dataset serevr")
parser.add_argument("-p", "--port", type=int, default=8088, help="the port of the dataset serevr")
parser.add_argument("-w", "--workers", type=int, default=4, help="the number of workers")
parser.add_argument("-m", "--module", type=str, default='wxbtool.specs.t850', help="module of a metrological model to load")
parser.add_argument("-s", "--setting", type=str, default='Setting', help="setting for a metrological model spec")
opt = parser.parse_args()


np.core.arrayprint._line_width = 150
np.set_printoptions(linewidth=np.inf)

early_stopping = 5
best = np.inf
count = 0

try:
    mdm = importlib.import_module(opt.module, package=None)
    setting = getattr(mdm, opt.setting)()
    spec = getattr(mdm, 'Spec')(setting)
except ImportError as e:
    print('failure when loading model')
    sys.exit(1)


time_str = arrow.now().format('YYYYMMDD_HHmmss')
model_path = Path(f'./dsserver/{time_str}')
model_path.mkdir(exist_ok=True, parents=True)
log_file = model_path / Path('dsserver.log')
logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info(str(opt))


app = Flask(__name__)
app.debug = False

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logger.addHandler(handler)

gunicorn_logger = logging.getLogger('gunicorn.info')
app.logger.handlers.extend(gunicorn_logger.handlers)
logger.handlers.extend(app.logger.handlers)

route = app.route

datasets = {}
spec.load_dataset('train', 'server')
spec.load_dataset('test', 'server')
dtrain = spec.dataset_train
deval = spec.dataset_eval
dtest = spec.dataset_test
datasets['train'] = dtrain
datasets['eval'] = deval
datasets['test'] = dtest


@route("/<string:hash>/<string:mode>")
def length(hash, mode):
    ds = datasets[mode]
    if ds.hashcode != hash:
        return flask.current_app.response_class('not found', status=404, mimetype="application/msgpack")

    logger.info('query length[%s] %d', mode, len(ds))
    msg = msgpack.dumps({
        'size': len(ds),
    })

    return flask.current_app.response_class(msg, status=200, mimetype="application/msgpack")


@route("/<string:hash>/<string:mode>/<int:idx>")
def seek(hash, mode, idx):
    ds = datasets[mode]
    if ds.hashcode != hash:
        return flask.current_app.response_class('not found', status=404, mimetype="application/msgpack")

    logger.info('query data[%s] at %d', mode, idx)
    inputs, targets = ds[idx]
    msg = msgpack.dumps({
        'inputs': inputs,
        'targets': targets,
    })

    return flask.current_app.response_class(msg, status=200, mimetype="application/msgpack")


def main():
    import gunicorn.app.base

    class StandaloneApplication(gunicorn.app.base.BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            config = {key: value for key, value in self.options.items()
                      if key in self.cfg.settings and value is not None}
            for key, value in config.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    print("PID %s" % str(os.getpid()))
    print("serving... %s" % opt.module)
    print("ip: %s" % opt.ip)
    print("port: %s" % opt.port)
    print("workers: %s" % opt.workers)

    options = {
        'bind': '%s:%s' % (opt.ip, opt.port),
        'workers': opt.workers,
    }
    StandaloneApplication(app, options).run()


if __name__ == "__main__":
    main()
