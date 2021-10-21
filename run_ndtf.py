from transformers import AdamW, RobertaForMultipleChoice, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
import torch
import sys
import argparse

from utils.engine import Engine
import os


def get_args(argv):
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-r', '--retrieval', choices=['IR', 'NSP', 'NN'],
                          help='retrieval solver for the contexts. Options: IR, NSP or NN', required=True)
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'],
                        help='device to train the model with. Options: cpu or gpu. Default: gpu')
    parser.add_argument('-p', '--pretrainings', default="",
                        help='path to the pretrainings model. Default: empty')
    parser.add_argument('-b', '--batchsize', default=8, type=int, help='size of the batches. Default: 8')
    parser.add_argument('-x', '--maxlen', default=64, type=int, help='max sequence length. Default: 64')
    parser.add_argument('-l', '--lr', default=1e-5, type=float, help='learning rate. Default: 1e-5')
    parser.add_argument('-e', '--epochs', default=2, type=int, help='number of epochs. Default: 2')
    parser.add_argument('-s', '--save', default=False, help='save model at the end of the training',
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    args = get_args(sys.argv[1:])
    print(args)
    engine = Engine(cfgs="")
    engine.args_phrase(args, "ndtf")
