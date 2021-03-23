from arghandler import *

from wxbtool.data.dsserver import main as dsmain
from wxbtool.nn.train import main as tnmain
from wxbtool.nn.test import main as ttmain


@subcmd
def help(parser, context, args):
    pass

@subcmd('dserve', help='start the dataset server')
def dserve(parser, context, args):
    parser.add_argument("-i", "--ip", type=str, default='127.0.0.1',
        help="the ip of the dataset serevr")
    parser.add_argument("-p", "--port", type=int, default=8088,
        help="the port of the dataset serevr")
    parser.add_argument("-w", "--workers", type=int, default=4,
        help="the number of workers")
    parser.add_argument("-m", "--module", type=str, default='wxbtool.specs.res5_625.t850weyn',
        help="module of a metrological model to load")
    parser.add_argument("-s", "--setting", type=str, default='Setting',
        help="setting for a metrological model spec")
    opt = parser.parse_args(args)

    dsmain(context, opt)


@subcmd('train', help='start training')
def train(parser, context, args):
    parser.add_argument("-g", "--gpu", type=str, default='0',
        help="index of gpu")
    parser.add_argument("-c", "--n_cpu", type=int, default=8,
        help="number of cpu threads to use during batch generation")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
        help="size of the batches")
    parser.add_argument("-e", "--epoch", type=int, default=0,
        help="current epoch to start training from")
    parser.add_argument("-n", "--n_epochs", type=int, default=200,
        help="number of epochs of training")
    parser.add_argument("-m", "--module", type=str, default='wxbtool.zoo.unet.t850d3',
        help="module of the metrological model to load")
    parser.add_argument("-l", "--load", type=str, default='',
        help="dump file of the metrological model to load")
    parser.add_argument("-k", "--check", type=str, default='',
        help="checkpoint file to load")
    parser.add_argument("-r", "--rate", type=float, default=0.001,
        help="learning rate")
    parser.add_argument("-w", "--weightdecay", type=float, default=0.0,
        help="weight decay")
    parser.add_argument("-d", "--data", type=str, default='',
        help="url of the dataset server")
    opt = parser.parse_args(args)

    tnmain(context, opt)


@subcmd('test', help='start testing')
def test(parser, context, args):
    parser.add_argument("-g", "--gpu", type=str, default='0',
        help="index of gpu")
    parser.add_argument("-c", "--n_cpu", type=int, default=8,
        help="number of cpu threads to use during batch generation")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
        help="size of the batches")
    parser.add_argument("-n", "--n_epochs", type=int, default=200,
        help="number of epochs of training")
    parser.add_argument("-m", "--module", type=str, default='wxbtool.zoo.unet.t850d3',
        help="module of the metrological model to load")
    parser.add_argument("-l", "--load", type=str, default='',
        help="dump file of the metrological model to load")
    parser.add_argument("-d", "--data", type=str, default='',
        help="url of the dataset server")
    opt = parser.parse_args(args)

    ttmain(context, opt)


def main():
    import sys

    handler = ArgumentHandler()
    if len(sys.argv) < 2:
        handler.run(['help'])
    else:
        handler.run(sys.argv[1:])
