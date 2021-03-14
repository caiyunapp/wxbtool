# -*- coding: utf-8 -*-

import os
import argparse
import importlib
import logging
import resource

from pathlib import Path

import arrow
import numpy as np
import torch as th
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau
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
parser.add_argument("-m", "--module", type=str, default='wxbtool.mdls.resunet', help="module of the metrological model to load")
parser.add_argument("-l", "--load", type=str, default='', help="dump file of the metrological model to load")
parser.add_argument("-k", "--check", type=str, default='', help="checkpoint file to load")
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

print('cudnn:', th.backends.cudnn.version())

np.core.arrayprint._line_width = 150
np.set_printoptions(linewidth=np.inf)

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


scheduler = None


def train_model(mdl):
    global scheduler
    optimizer = th.optim.Adam(mdl.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    try:
        if opt.load != '':
            dump = th.load(opt.load, map_location='cpu')
            mdm.model.load_state_dict(dump)
    except ImportError as e:
        logger.exception(e)
        exit(1)

    if th.cuda.is_available():
        mdl = mdl.cuda()

    def train(epoch, mdl):
        mdl.train()
        dataloader = DataLoader(mdm.model.dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
        loss_per_epoch = 0.0
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

            optimizer.zero_grad()
            results = mdl(*[], **inputs)
            loss = mdm.model.lossfun(inputs, results, targets)
            loss.backward()
            optimizer.step()

            logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()}')

            loss_per_epoch += loss.item() * list(results.values())[0].size()[0]

        logger.info(f'Epoch: {epoch + 1:03d} | Train Loss: {loss_per_epoch / mdm.model.train_size}')

        # evaluation
        loss_eval = evaluate(epoch)
        logger.info(f'Epoch: {epoch + 1:03d} | Eval loss: {loss_eval}')
        scheduler.step(loss_eval)

        global best, count
        if loss_eval.item() >= best:
            count += 1
        else:
            count = 0
            best = loss_eval

        if count == early_stopping:
            logger.info('early stopping reached, best loss is {:5f}'.format(best))

    def evaluate(epoch):
        mdl.eval()
        dataloader = DataLoader(mdm.model.dataset_eval, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
        loss_per_epoch = 0.0
        rmse_per_epoch_t = 0.0
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
                logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()}')
                loss_per_epoch += loss.item() * list(results.values())[0].size()[0]

                _, tgt = mdm.model.get_targets(**targets)
                _, rst = mdm.model.get_results(**results)
                tgt = tgt.detach().cpu().numpy().reshape(-1, 1, 32, 64)
                rst = rst.detach().cpu().numpy().reshape(-1, 1, 32, 64)
                rmse = np.sqrt(np.mean((rst - tgt) * (rst - tgt)))
                logger.info(
                    f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()} | Temperature RMSE: {rmse}')
                rmse_per_epoch_t += np.nan_to_num(rmse * list(results.values())[0].size()[0])

        rmse_total = rmse_per_epoch_t / mdm.model.eval_size
        logger.info(f'Epoch: {epoch + 1:03d} | Eval Loss: {loss_per_epoch / mdm.model.eval_size}')
        logger.info(f'Epoch: {epoch + 1:03d} | Eval RMSE: {rmse_total}')

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

        th.save(mdl.state_dict(), model_path / f'm_rmse{rmse_total:0.8f}_epoch{epoch + 1:03d}.mdl')
        glb = list(model_path.glob('*.mdl'))
        if len(glb) > 6:
            for p in sorted(glb)[-3:]:
                os.unlink(p)

        th.save({
            'net': mdl.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_path / f'z_epoch{epoch + 1:03d}.chk')
        glb = list(model_path.glob('*.chk'))
        if len(glb) > 1:
            os.unlink(sorted(glb)[0])

        return rmse_total

    try:
        for epoch in range(opt.n_epochs):
            train(epoch, mdl)
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    if mdm != None:
        mdm.model.load_dataset('train')
        if len(opt.gpu.split(',')) > 0:
            train_model(nn.DataParallel(mdm.model, output_device=0))
        else:
            train_model(mdm.model)
    print('Training Finished!')
