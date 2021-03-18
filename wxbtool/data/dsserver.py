# -*- coding: utf-8 -*-

import argparse
import importlib
import logging
import resource
import arrow
import sys
import os

import numpy as np
import flask
import msgpack
import msgpack_numpy as m
m.patch()

from pathlib import Path
from flask import Flask


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", type=int, default=8088, help="the port of the dataset serevr")
parser.add_argument("-s", "--spec", type=str, default='wxbtool.specs.t850', help="spec of a metrological model to load")
opt = parser.parse_args()

np.core.arrayprint._line_width = 150
np.set_printoptions(linewidth=np.inf)

name = opt.model
early_stopping = 5
best = np.inf
count = 0

mdm = None
try:
    mdm = importlib.import_module(opt.module, package=None)
except ImportError as e:
    print('failure when loading model')
    sys.exit(1)

name = mdm.model.name
time_str = arrow.now().format('YYYYMMDD_HHmmss')
model_path = Path(f'./dsserver/{name}-{time_str}')
model_path.mkdir(exist_ok=True, parents=True)
log_file = model_path / Path('dsserver.log')

logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info(str(opt))


def setup(appname):
    application = Flask(appname)
    logger = logging.getLogger(appname)
    application.debug = False
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return application


app = setup(__name__)
route = app.route

datasets = {}


@route("/<str:mode>/<int:idx>")
def seek(mode, idx):
    inputs, targets = datasets[mode][idx]
    msg = msgpack.dumps({
        'inputs': inputs,
        'targets': targets,
    })
    return flask.current_app.response_class(msg, status=200, mimetype="application/msgpack")


def main():
    mdm.model.load_dataset('train')
    mdm.model.load_dataset('test')
    dtrain = mdm.model.dataset_train
    deval = mdm.model.dataset_eval
    dtest = mdm.model.dataset_test
    datasets['train'] = dtrain
    datasets['eval'] = deval
    datasets['test'] = dtest

    print("PID %s" % str(os.getpid()))
    print("serving... %s" % name)
    app.debug = False
    app.run(host="0.0.0.0", port=opt.port, debug=False)


if __name__ == "__main__":
    main()
