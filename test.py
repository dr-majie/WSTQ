from transformers import RobertaTokenizer
import numpy as np
import json
from tqdm import tqdm
import torch
import random
import sys
import argparse
from utils.engine import Engine


def get_args(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'],
                        help='device to train the model with. Options: cpu or gpu. Default: gpu')
    parser.add_argument('-x', '--maxlen', default=180, type=int, help='max sequence length. Default: 180')
    parser.add_argument('-b', '--batchsize', default=1, type=int, help='size of the batches. Default: 1')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-r', '--retrieval', choices=['IR', 'NSP', 'NN'],
                          help='retrieval solver for the contexts. Options: IR, NSP or NN', required=True)
    required.add_argument('-p', '--pretrainings', required=True)
    required.add_argument('-m', '--model_type', choices=['dmc', 'ndmc', 'ndtf'], required=True)
    args = parser.parse_args()
    #print(args)
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
    engine.arg_phrase_test(args)
