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
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import matplotlib.colors

from utils import *

GLOVE_PATH = './Infersent/dataset/GloVe/glove.840B.300d.txt'
INFERSENT_PRETRAINED_PATH = './Infersent/encoder/infersent.allnli.pickle'

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

tableau20 = sorted(tableau20, key=lambda x: x[0])

sys.path.append('./Infersent/')

model = torch.load(INFERSENT_PRETRAINED_PATH, map_location=lambda storage, loc: storage)

model.set_glove_path(GLOVE_PATH)

model.build_vocab_k_words(K=100000)
def calculate_real_gain(real_inputs, real_outputs, sampling_size):
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

        inputs1_embeddings = model.encode(inputs1, bsize=128, tokenize=False, verbose=True)
        inputs2_embeddings = model.encode(inputs2, bsize=128, tokenize=False, verbose=True)

        for in1, in2, out1, out2 in zip(inputs1_embeddings, inputs2_embeddings, outputs1, outputs2):
            input_distance = cosine(in1, in2)
            output_distance = jsd(out1, out2)
            gain = np.clip(output_distance / (input_distance + EPS), -20., 20.)
            gains.append(gain)

    # calculate bootstrap estimates for the mean and standard deviation# calcu
    mean_results = bs.bootstrap(gains, stat_func=bs_stats.mean)

    # see advanced_bootstrap_features.ipynb for a discussion of how to use the stat_func arg
    stdev_results = bs.bootstrap(gains, stat_func=bs_stats.std)
    return mean_results, stdev_results

def calculate_adversarial_gain(real_inputs, real_outputs, adv_inputs, adv_outputs):
    input_distances = []
    output_distances = []
    gains = []

    inputs1_embeddings = model.encode(real_inputs, bsize=128, tokenize=False, verbose=True)
    inputs2_embeddings = model.encode(adv_inputs, bsize=128, tokenize=False, verbose=True)

    for in1, in2, out1, out2 in zip(inputs1_embeddings, inputs2_embeddings, real_outputs, adv_outputs):
        input_distance = cosine(in1, in2)
        output_distance = jsd(out1, out2)
        input_distances.append(input_distance)
        output_distances.append(output_distance)
        gain = np.clip(output_distance / (input_distance + EPS), -20., 20.)
        gains.append(gain)

    return input_distances, output_distances, gains


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
    parser.add_argument('--save', default=False, action="store_true")
    args = parser.parse_args(arguments)

    print(args)

    lines = args.infile.readlines()

    reals = f7(lines[0:][::2])
    adversarials = f7(lines[1:][::2])

    real_ins, real_outs = process_lines(reals)
    adv_ins, adv_outs = process_lines(adversarials)

    input_distances, output_distances, gains = calculate_adversarial_gain(real_ins, real_outs, adv_ins, adv_outs)

    most_gain_samples = np.array(gains).argsort()[-5:][::-1]
    least_gain_samples = np.array(gains).argsort()[:5][::-1]
    print("=======================BEST=====================")
    for x in most_gain_samples:
        print("Original Input: {}".format(real_ins[x]))
        print("Original SoftMax Output: {}".format(real_outs[x]))
        print("Adversarial Input: {}".format(adv_ins[x]))
        print("Adversarial SoftMax Output: {}".format(adv_outs[x]))
        print("Gain: {}".format(gains[x]))
        print("Input Distance: {}".format(input_distances[x]))
        print("Output Distance: {}".format(output_distances[x]))
    print("=======================WORST=====================")
    for x in least_gain_samples:
        print("Original Input: {}".format(real_ins[x]))
        print("Original SoftMax Output: {}".format(real_outs[x]))
        print("Adversarial Input: {}".format(adv_ins[x]))
        print("Adversarial SoftMax Output: {}".format(adv_outs[x]))
        print("Gain: {}".format(gains[x]))
        print("Input Distance: {}".format(input_distances[x]))
        print("Output Distance: {}".format(output_distances[x]))
    gains = gains
    gains = normalize_array(np.clip(np.array(gains), -100, 100))
    sizes = [np.pi*(4.0 + x*10)**2 for x in gains]
    # cmap = plt.cm.rainbow
    cmap = plt.get_cmap('inferno')
    norm = matplotlib.colors.Normalize(vmax=.9)
    colors = cmap(norm(gains))
    fig = plt.figure(figsize=(16, 8))

    axis_font = {'fontname':'Arial', 'size':'32'}
    plt.title("HotFlip Adversary on SST2", **axis_font)
    plt.xlabel("Input Distance" + "\n" + "(InferSent-Cosine)", **axis_font)
    plt.ylabel(r"Output Distance" + "\n" + "(Jensen-Shannon divergence)", **axis_font)
    plt.scatter(input_distances, output_distances, s=sizes, c=colors, alpha=0.65, edgecolors='none')
    if args.save:
        fig.savefig('sentiment_adversary.pdf', dpi=fig.dpi, bbox_inches='tight')
    else:
        plt.show()



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
