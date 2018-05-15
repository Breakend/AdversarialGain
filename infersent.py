#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse

from random import randint
import matplotlib
import random
import numpy as np
import torch
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
GLOVE_PATH = './Infersent/dataset/GloVe/glove.840B.300d.txt'
INFERSENT_PRETRAINED_PATH = './Infersent/encoder/infersent.allnli.pickle'

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def main(arguments):
    sys.path.append('./Infersent/')

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('outfile', help="Input file", type=argparse.FileType('r'))
    # parser.add_argument('-o', '--outfile', help="Output file",
    #                     default=sys.stdout, type=argparse.FileType('w'))
    parser.add_argument('--sampling_size', type=int, default=10000)
    args = parser.parse_args(arguments)

    model = torch.load(INFERSENT_PRETRAINED_PATH, map_location=lambda storage, loc: storage)
    model.set_glove_path(GLOVE_PATH)
    model.build_vocab_k_words(K=100000)

    inputs = []
    for line in args.infile:
        l = line.strip()
        if l: inputs.append(l)

    outputs = []
    for line in args.outfile:
        l = line.strip()
        if l: outputs.append(l)

    assert len(inputs) == len(outputs)
    gains = []
    while len(gains) < args.sampling_size:
        subsampling_idxs = random.sample(range(len(outputs)), min(args.sampling_size*2,len(outputs))) #numpy.random.randint(0, len(outputs), (args.sampling_size*2,))

        if len(subsampling_idxs) % 2 != 0:
            subsampling_idxs = subsampling_idxs[1:]

        batch1 = subsampling_idxs[:int(len(subsampling_idxs)/2)]
        batch2 = subsampling_idxs[int(len(subsampling_idxs)/2):]
        inputs1 = [inputs[i] for i in batch1]
        inputs2 = [inputs[i] for i in batch2]
        outputs1 = [outputs[i] for i in batch1]
        outputs2 = [outputs[i] for i in batch2]

        inputs1_embeddings = model.encode(inputs1, bsize=128, tokenize=False, verbose=True)
        inputs2_embeddings = model.encode(inputs2, bsize=128, tokenize=False, verbose=True)
        outputs1_embeddings = model.encode(outputs1, bsize=128, tokenize=False, verbose=True)
        outputs2_embeddings = model.encode(outputs2, bsize=128, tokenize=False, verbose=True)

        for in1, in2, out1, out2 in zip(inputs1_embeddings, inputs2_embeddings, outputs1_embeddings, outputs2_embeddings):
            gain = cosine(out1, out2) / cosine(in1, in2)
            gains.append(gain)

    gains = np.array(gains)
    # calculate bootstrap estimates for the mean and standard deviation# calcu
    mean_results = bs.bootstrap(gains, stat_func=bs_stats.mean)

    # see advanced_bootstrap_features.ipynb for a discussion of how to use the stat_func arg
    stdev_results = bs.bootstrap(gains, stat_func=bs_stats.std)

    print(('Bootstrapped mean'))
    print('\t' + str(mean_results))
    print('')
    print('Bootstrapped stdev')
    print('\t' + str(stdev_results))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
