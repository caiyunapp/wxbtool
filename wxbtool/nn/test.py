# -*- coding: utf-8 -*-

import importlib
import logging
import os
import resource
from pathlib import Path

import arrow
import numpy as np
import sys
import torch as th
from torch.utils.data import DataLoader

from wxbtool.util.plotter import plot

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

print('cudnn:', th.backends.cudnn.version())

np.core.arrayprint._line_width = 150
np.set_printoptions(linewidth=np.inf)

early_stopping = 5
best = np.inf
count = 0


def test_model(opt, mdl, logger=None):
    try:
        if opt.load != '':
            dump = th.load(opt.load, map_location='cpu')
            mdl.load_state_dict(dump)
    except ImportError as e:
        logger.exception(e)
        raise e

    if th.cuda.is_available():
        mdl = mdl.cuda()

    def test(mdl):
        mdl.eval()

        dataloader = DataLoader(mdl.dataset_test, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
        loss_per_epoch = 0.0
        rmse_per_epoch_t = 0.0
        inputs, targets, results = None, None, None
        for step, sample in enumerate(dataloader):
            inputs, targets = sample

            inputs = {
                v: th.as_tensor(inputs[v], dtype=th.float32) for v in mdl.setting.vars
            }
            targets = {
                v: th.as_tensor(targets[v], dtype=th.float32) for v in mdl.setting.vars
            }

            if th.cuda.is_available():
                inputs = {
                    v: inputs[v].cuda() for v in mdl.setting.vars
                }
                targets = {
                    v: targets[v].cuda() for v in mdl.setting.vars
                }
                mdl.constant = mdl.constant.cuda()
                mdl.weight = mdl.weight.cuda()

            with th.no_grad():
                results = mdl(*[], **inputs)
                loss = mdl.lossfun(inputs, results, targets)
                logger.info(f'Step: {step + 1:03d} | Loss: {loss.item()}')
                loss_per_epoch += loss.item() * list(results.values())[0].size()[0]

                _, tgt = mdl.get_targets(**targets)
                _, rst = mdl.get_results(**results)
                tgt = tgt.detach().cpu().numpy().reshape(-1, 1, 32, 64)
                rst = rst.detach().cpu().numpy().reshape(-1, 1, 32, 64)
                mdl.eva.weight = mdl.eva.weight.reshape(1, 1, 32, 64)
                mdl.eva.weight = mdl.eva.weight * (mdl.eva.weight > 0)
                rmse = mdl.eva.weighted_rmse(rst, tgt)
                logger.info(f'Step: {step + 1:03d} | Loss: {loss.item()} | Temperature RMSE: {rmse}')
                rmse_per_epoch_t += np.nan_to_num(rmse * list(results.values())[0].size()[0])

        rmse_total = rmse_per_epoch_t / mdl.test_size
        logger.info(f'Test Loss: {loss_per_epoch / mdl.test_size}')
        logger.info(f'Test RMSE: {rmse_total}')

        vars_in, _ = mdl.get_inputs(**inputs)
        for bas, var in enumerate(mdl.setting.vars_in):
            for ix in range(mdl.setting.input_span):
                img = vars_in[var][0, ix].detach().cpu().numpy().reshape(32, 64)
                plot(var, open('%s_inp_%d.png' % (var, ix), mode='wb'), img)

        vars_fc, _ = mdl.get_results(**results)
        vars_tg, _ = mdl.get_targets(**targets)
        for bas, var in enumerate(mdl.setting.vars_out):
            fcst = vars_fc[var][0].detach().cpu().numpy().reshape(32, 64)
            tgrt = vars_tg[var][0].detach().cpu().numpy().reshape(32, 64)
            plot(var, open('%s_fcs.png' % var, mode='wb'), fcst)
            plot(var, open('%s_tgt.png' % var, mode='wb'), tgrt)

    try:
        test(mdl)
    except Exception as e:
        print(e)
        logger.exception(e)


def main(context, opt):
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
        sys.path.insert(0, os.getcwd())
        mdm = importlib.import_module(opt.module, package=None)

        name = mdm.model.name
        time_str = arrow.now().format('YYYYMMDD_HHmmss')
        model_path = Path(f'./trains/{name}-{time_str}')
        model_path.mkdir(exist_ok=True, parents=True)
        log_file = model_path / Path('train.log')

        logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.info(str(opt))
        if mdm != None:
            if opt.data != '':
                mdm.model.load_dataset('test', 'client', url=opt.data)
            else:
                mdm.model.load_dataset('test', 'server')
            test_model(opt, mdm.model, logger=logger)
        print('Test Finished!')
    except ImportError as e:
        exc_info = sys.exc_info()
        print(e)
        print('failure when loading model')
        import traceback
        traceback.print_exception(*exc_info)
        del exc_info
        sys.exit(-1)
