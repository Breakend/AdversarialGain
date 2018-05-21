#!/usr/bin/env python

"""A simple python script template.

"""

from __future__ import print_function
import os
import sys
import argparse
import torch
import numpy as np
import random
from utils import *
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
GLOVE_PATH = './Infersent/dataset/GloVe/glove.840B.300d.txt'
INFERSENT_PRETRAINED_PATH = './Infersent/encoder/infersent.allnli.pickle'


sys.path.append('./Infersent/')

model = torch.load(INFERSENT_PRETRAINED_PATH, map_location=lambda storage, loc: storage)

model.set_glove_path(GLOVE_PATH)

model.build_vocab_k_words(K=100000)

def find_num_different_words(one, two):
    one = one.split(" ")
    two = two.split(" ")
    diffs = 0
    for x,y in zip(one, two):
        if x != y:
            diffs += 1
    return diffs

def hinge(one, two):
    if np.argmax(one) != np.argmax(two):
        return 1.0
    else:
        return 0.0

def distance(in1, in2, distance_type):
    if distance_type == "infersent-cosine":
        return cosine(in1, in2)
    elif distance_type == "word-difference":
        return find_num_different_words(in1, in2)
    elif distance_type == "js":
        return jsd(in1, in2)
    elif distance_type == "hinge":
        return hinge(one, two)
    else:
        raise NotImplementedError

def calculate_real_gain(real_inputs, real_outputs, sampling_size, input_distance_type, output_distance_type):
    gains = []
    while len(gains) < sampling_size:
        subsampling_idxs = random.sample(range(len(real_outputs)), min(sampling_size*2,len(real_outputs))) #numpy.random.randint(0, len(outputs), (args.sampling_size*2,))

        if len(subsampling_idxs) % 2 != 0:
            subsampling_idxs = subsampling_idxs[1:]

        batch1 = subsampling_idxs[:int(len(subsampling_idxs)/2)]
        batch2 = subsampling_idxs[int(len(subsampling_idxs)/2):]
        inputs1 = [real_inputs[i] for i in batch1]
        inputs2 = [real_inputs[i] for i in batch2]
        outputs1 = [real_outputs[i] for i in batch1]
        outputs2 = [real_outputs[i] for i in batch2]

        if input_distance_type == "infersent-cosine":
            inputs1 = model.encode(inputs1, bsize=128, tokenize=False, verbose=True)
            inputs2 = model.encode(inputs2, bsize=128, tokenize=False, verbose=True)

        for in1, in2, out1, out2 in zip(inputs1, inputs2, outputs1, outputs2):
            input_distance = distance(in1, in2, distance_type=input_distance_type)
            output_distance = distance(out1, out2, distance_type=output_distance_type)
            gain = np.clip(output_distance / (input_distance + EPS), -20., 20.)
            gains.append(gain)

    gains = np.array(gains)
    # calculate bootstrap estimates for the mean and standard deviation# calcu
    mean_results = bs.bootstrap(gains, stat_func=bs_stats.mean)

    # see advanced_bootstrap_features.ipynb for a discussion of how to use the stat_func arg
    stdev_results = bs.bootstrap(gains, stat_func=bs_stats.std)
    return mean_results, stdev_results

def calculate_adversarial_gain(inputs1, inputs2, outputs1, outputs2, input_distance_type, output_distance_type):
    gains = []
    if input_distance_type == "infersent-cosine":
        inputs1 = model.encode(inputs1, bsize=128, tokenize=False, verbose=True)
        inputs2 = model.encode(inputs2, bsize=128, tokenize=False, verbose=True)

    for in1, in2, out1, out2 in zip(inputs1, inputs2, outputs1, outputs2):
        input_distance = distance(in1, in2, distance_type=input_distance_type)
        output_distance = distance(out1, out2, distance_type=output_distance_type)
        gain = np.clip(output_distance / (input_distance + EPS), -20., 20.)
        gains.append(gain)

    gains = np.array(gains)
    # calculate bootstrap estimates for the mean and standard deviation# calcu
    mean_results = bs.bootstrap(gains, stat_func=bs_stats.mean)

    # see advanced_bootstrap_features.ipynb for a discussion of how to use the stat_func arg
    stdev_results = bs.bootstrap(gains, stat_func=bs_stats.std)
    return mean_results, stdev_results


def process_lines(lines):
    inputs, outputs = [], []
    for l in lines:
        if l:
            distribution, sentence = l.split('\t')
            sentence = sentence.strip()
            inputs.append(sentence)
            distribution = np.fromstring(distribution.strip("[").strip("]").strip(), sep = ' ')
            outputs.append(distribution)
    return inputs, outputs

def main(arguments):


    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('--output_distance_type', type=str, default="js")
    parser.add_argument('--input_distance_type', type=str, default="infersent-cosine")

    args = parser.parse_args(arguments)

    print(args)

    lines = args.infile.readlines()

    reals = f7(lines[0:][::2])
    adversarials = f7(lines[1:][::2])

    real_ins, real_outs = process_lines(reals)
    adv_ins, adv_outs = process_lines(adversarials)
    gain_boostrap_mean, gain_boostrap_std = calculate_real_gain(real_ins, real_outs, 1000, args.input_distance_type, args.output_distance_type)
    print("Real bootstrap region: {} +/- {}".format(gain_boostrap_mean, gain_boostrap_std))

    adversarial_boostrap_mean, adversarial_boostrap_std = calculate_adversarial_gain(real_ins, real_outs, adv_ins, adv_outs, args.input_distance_type, args.output_distance_type)

    print("Adversarial Bootstrap region: {} +/ {}".format(adversarial_boostrap_mean, adversarial_boostrap_std))
    # TODO: generate graph from some real samples


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
