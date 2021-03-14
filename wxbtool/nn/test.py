# -*- coding: utf-8 -*-

import argparse
import importlib
import logging
import os
import resource
from pathlib import Path

import arrow
import numpy as np
import torch as th
from torch.utils.data import DataLoader

from wxbtool.plot.plotter import plot

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=str, default='0', help="index of gpu")
parser.add_argument("-c", "--n_cpu", type=int, default=64, help="number of cpu threads to use during batch generation")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("-e", "--epoch", type=int, default=0, help="current epoch to start training from")
parser.add_argument("-n", "--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("-m", "--model", type=str, default='', help="metrological model to load")
parser.add_argument("-k", "--check", type=str, default='', help="checkpoint file to load")
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

print('cudnn:', th.backends.cudnn.version())

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
    exit(1)

name = mdm.model.setting.name
time_str = arrow.now().format('YYYYMMDD_HHmmss')
model_path = Path(f'./trains/{name}-{time_str}')
model_path.mkdir(exist_ok=True, parents=True)
log_file = model_path / Path('train.log')

logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info(str(opt))


def test_model(mdl):
    try:
        if opt.check != '':
            checkpoint = th.load(opt.check, map_location='cpu')
            mdm.model.load_state_dict(checkpoint)
    except ImportError as e:
        logger.exception(e)
        exit(1)

    if th.cuda.is_available():
        mdl = mdl.cuda()

    def test(mdl):
        mdl.eval()

        dataloader = DataLoader(mdm.model.dataset_test, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
        loss_per_epoch = 0.0
        rmse_per_epoch_t = 0.0
        inputs, targets, results = None, None, None
        for step, sample in enumerate(dataloader):
            inputs, targets = sample

            inputs = {
                v: th.as_tensor(np.copy(inputs[v]), dtype=th.float32) for v in mdm.setting.vars
            }
            targets = {
                v: th.as_tensor(np.copy(targets[v]), dtype=th.float32) for v in mdm.setting.vars
            }

            if th.cuda.is_available():
                inputs = {
                    v: inputs[v].cuda() for v in mdm.setting.vars
                }
                targets = {
                    v: targets[v].cuda() for v in mdm.setting.vars
                }
                mdm.model.constant = mdm.model.constant.cuda()
                mdm.model.weight = mdm.model.weight.cuda()

            with th.no_grad():
                results = mdl(*[], **inputs)
                loss = mdm.model.lossfun(inputs, results, targets)
                logger.info(f'Step: {step + 1:03d} | Loss: {loss.item()}')
                loss_per_epoch += loss.item() * list(results.values())[0].size()[0]

                _, tgt = mdm.model.get_targets(**targets)
                _, rst = mdm.model.get_results(**results)
                tgt = tgt.detach().cpu().numpy().reshape(-1, 1, 32, 64)
                rst = rst.detach().cpu().numpy().reshape(-1, 1, 32, 64)
                mdm.model.eva.weight = mdm.model.eva.weight.reshape(1, 1, 32, 64)
                mdm.model.eva.weight = mdm.model.eva.weight * (mdm.model.eva.weight > 0)
                rmse = mdm.model.eva.weighted_rmse(rst, tgt)
                logger.info(f'Step: {step + 1:03d} | Loss: {loss.item()} | Temperature RMSE: {rmse}')
                rmse_per_epoch_t += np.nan_to_num(rmse * list(results.values())[0].size()[0])

        rmse_total = rmse_per_epoch_t / mdm.model.test_size
        logger.info(f'Test Loss: {loss_per_epoch / mdm.model.test_size}')
        logger.info(f'Test RMSE: {rmse_total}')

        vars_in, _ = mdm.model.get_inputs(**inputs)
        for bas, var in enumerate(mdm.setting.vars_in):
            for ix in range(mdm.setting.input_span):
                img = vars_in[var][0, ix].detach().cpu().numpy().reshape(32, 64)
                plot(var, open('%s_inp_%d.png' % (var, ix), mode='wb'), img)

        vars_fc, _ = mdm.model.get_results(**results)
        vars_tg, _ = mdm.model.get_targets(**targets)
        for bas, var in enumerate(mdm.setting.vars_out):
            fcst = vars_fc[var][0].detach().cpu().numpy().reshape(32, 64)
            tgrt = vars_tg[var][0].detach().cpu().numpy().reshape(32, 64)
            plot(var, open('%s_fcs.png' % var, mode='wb'), fcst)
            plot(var, open('%s_tgt.png' % var, mode='wb'), tgrt)

    try:
        test(mdl)
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    if mdm != None:
        mdm.model.load_dataset('test')
        test_model(mdm.model)
    print('Test Finished!')
