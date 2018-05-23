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
import matplotlib.pyplot as plt
import matplotlib.colors
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
GLOVE_PATH = './Infersent/dataset/GloVe/glove.840B.300d.txt'
INFERSENT_PRETRAINED_PATH = './Infersent/encoder/infersent.allnli.pickle'


sys.path.append('./Infersent/')

model = torch.load(INFERSENT_PRETRAINED_PATH, map_location=lambda storage, loc: storage)

model.set_glove_path(GLOVE_PATH)

model.build_vocab_k_words(K=100000)



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
    parser.add_argument('original_input', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('--seq2sick', default=False, action="store_true")
    parser.add_argument('--wordadver', default=False, action="store_true")
    parser.add_argument('--extra_input', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('--output_distance_type', type=str, default="infersent-cosine")
    parser.add_argument('--input_distance_type', type=str, default="infersent-cosine")
    parser.add_argument('--save', default=False, action="store_true")
    parser.add_argument('--plot', default=False, action="store_true")
    parser.add_argument('--get_bootstrap', default=False, action="store_true")
    parser.add_argument('--get_samples', default=False, action="store_true")

    args = parser.parse_args(arguments)

    print(args)
    if args.seq2sick:
        lines = args.original_input.readlines()
        seq2sick_lines = args.extra_input.readlines()
        real_ins = []
        real_outs = []
        adv_ins = []
        adv_outs = []
        for line, seq2sickline in zip(lines, seq2sick_lines):
            adversarial_input, adversarial_output, normal_output =seq2sickline.split("\t")
            adversarial_input = adversarial_input.strip()
            adversarial_output = adversarial_output.strip()
            normal_output = normal_output.strip()
            line = line.strip()
            adv_ins.append(adversarial_input)
            adv_outs.append(adversarial_output)
            real_outs.append(normal_output)
            real_ins.append(line)
    elif args.wordadver:
        lines = args.original_input.readlines()

        reals = lines[0:][::2]
        adversarials = lines[1:][::2]

        real_ins, real_outs = process_lines(reals)
        adv_ins, adv_outs = process_lines(adversarials)
    else:
        raise NotImplementedError

    if args.get_bootstrap:
        gain_boostrap_mean, gain_boostrap_std = calculate_real_gain(real_ins, real_outs, 1000, args.input_distance_type, args.output_distance_type, model)
        print("Real bootstrap region: {} +/- {}".format(gain_boostrap_mean, gain_boostrap_std))

        adversarial_boostrap_mean, adversarial_boostrap_std = calculate_adversarial_gain(real_ins, real_outs, adv_ins, adv_outs, args.input_distance_type, args.output_distance_type, model)

        print("Adversarial Bootstrap region: {} +/ {}".format(adversarial_boostrap_mean, adversarial_boostrap_std))
    # TODO: generate graph from some real samples

    if args.plot:
        input_distances, output_distances, gains = calculate_adversarial_gain_details(real_ins, real_outs, adv_ins, adv_outs, args.input_distance_type, args.output_distance_type, model)

        gains = gains
        gains = normalize_array(np.clip(np.array(gains), -100, 100))
        sizes = [np.pi*(4.0 + x*10)**2 for x in gains]
        # cmap = plt.cm.rainbow

        cmap = plt.get_cmap('inferno')
        norm = matplotlib.colors.Normalize()
        colors = cmap(norm(gains))
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.xlim(xmax=.08)
        plt.ylim(ymax=1.01, ymin=-.01)
        axis_font = {'fontname':'Arial', 'size':'32'}
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname(axis_font['fontname'])
            label.set_fontsize(axis_font['size'])
        plt.title("Seq2Sick Adversary on GigaWord", **axis_font)
        plt.xlabel("Input Distance" + "\n" + "(InferSent-Cosine)", **axis_font)
        plt.ylabel(r"Output Distance" + "\n" + "(InferSent-Cosine)", **axis_font)
        plt.scatter(input_distances, output_distances, s=sizes, c=colors, alpha=0.65, edgecolors='none')
        if args.save:
            fig.savefig('seq2sick_adversary.pdf', dpi=fig.dpi, bbox_inches='tight')
        else:
            plt.show()

    if args.get_samples:
        input_distances, output_distances, gains = calculate_adversarial_gain_details(real_ins, real_outs, adv_ins, adv_outs, args.input_distance_type, args.output_distance_type, model)
        import tsv
        most_gain_samples = np.array(gains).argsort()[:][::-1]
        writer = tsv.TsvWriter(open("samples.tsv", "w"))
        writer.list_line(["input", "output", "adv_input", "adv_output", "d_in", "d_out", "gain"])
        for z in most_gain_samples:
            col = (real_ins[z], real_outs[z], adv_ins[z], adv_outs[z], input_distances[z], output_distances[z], gains[z])
            writer.list_line(col)

        writer.close()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
