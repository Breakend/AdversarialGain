import numpy as np
import scipy as sp
import random
import scipy.stats
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
EPS = 1e-4

def cosine(u, v):
    return max(1.0 - np.abs((np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))), 0.0)

def jsd(p, q, base=np.e):
    '''
        Implementation of pairwise `jsd` based on
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    p = np.asarray(p)
    q = np.asarray(q)
    m = 1./2*(p + q)
    return sp.stats.entropy(p,m, base=base)/2. +  sp.stats.entropy(q, m, base=base)/2.

def normalize_array(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def find_num_different_words(one, two):
    one = one.split(" ")
    two = two.split(" ")
    maxxlen = max(len(one), len(two))
    diffs = 0
    for i in range(maxxlen):
        if i >= len(one):
            if two[i]: diffs += 1
            continue
        if i >= len(two):
            if one[i]: diffs += 1
            continue
        if one[i] and two[i] and two[i] != one[i]:
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
        return hinge(in1, in2)
    else:
        raise NotImplementedError





def calculate_real_gain(real_inputs, real_outputs, sampling_size, input_distance_type, output_distance_type, model):
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

        if output_distance_type == "infersent-cosine":
            outputs1 = model.encode(outputs1, bsize=128, tokenize=False, verbose=True)
            outputs2 = model.encode(outputs2, bsize=128, tokenize=False, verbose=True)

        for in1, in2, out1, out2 in zip(inputs1, inputs2, outputs1, outputs2):
            input_distance = distance(in1, in2, distance_type=input_distance_type)
            output_distance = distance(out1, out2, distance_type=output_distance_type)
            gain = output_distance / (input_distance + EPS)
            gains.append(gain)

    gains = np.array(gains)
    # calculate bootstrap estimates for the mean and standard deviation# calcu
    mean_results = bs.bootstrap(gains, stat_func=bs_stats.mean)

    # see advanced_bootstrap_features.ipynb for a discussion of how to use the stat_func arg
    stdev_results = bs.bootstrap(gains, stat_func=bs_stats.std)
    return mean_results, stdev_results

def calculate_adversarial_gain(inputs1, outputs1, inputs2, outputs2, input_distance_type, output_distance_type, model):
    gains = []
    if input_distance_type == "infersent-cosine":
        inputs1 = model.encode(inputs1, bsize=128, tokenize=False, verbose=True)
        inputs2 = model.encode(inputs2, bsize=128, tokenize=False, verbose=True)
    if output_distance_type == "infersent-cosine":
        outputs1 = model.encode(outputs1, bsize=128, tokenize=False, verbose=True)
        outputs2 = model.encode(outputs2, bsize=128, tokenize=False, verbose=True)

    for in1, in2, out1, out2 in zip(inputs1, inputs2, outputs1, outputs2):
        input_distance = distance(in1, in2, distance_type=input_distance_type)
        output_distance = distance(out1, out2, distance_type=output_distance_type)
        gain = output_distance / (input_distance + EPS)
        gains.append(gain)

    gains = np.array(gains)
    # calculate bootstrap estimates for the mean and standard deviation# calcu
    mean_results = bs.bootstrap(gains, stat_func=bs_stats.mean)

    # see advanced_bootstrap_features.ipynb for a discussion of how to use the stat_func arg
    stdev_results = bs.bootstrap(gains, stat_func=bs_stats.std)
    return mean_results, stdev_results


def calculate_adversarial_gain_details(inputs1, outputs1, inputs2, outputs2, input_distance_type, output_distance_type, model):
    input_distances = []
    output_distances = []
    gains = []
    if input_distance_type == "infersent-cosine":
        inputs1 = model.encode(inputs1, bsize=128, tokenize=False, verbose=True)
        inputs2 = model.encode(inputs2, bsize=128, tokenize=False, verbose=True)
    if output_distance_type == "infersent-cosine":
        outputs1 = model.encode(outputs1, bsize=128, tokenize=False, verbose=True)
        outputs2 = model.encode(outputs2, bsize=128, tokenize=False, verbose=True)

    for in1, in2, out1, out2 in zip(inputs1, inputs2, outputs1, outputs2):
        input_distance = distance(in1, in2, distance_type=input_distance_type)
        output_distance = distance(out1, out2, distance_type=output_distance_type)
        input_distances.append(input_distance)
        output_distances.append(output_distance)
        gain = output_distance / (input_distance + EPS)
        gains.append(gain)

    return input_distances, output_distances, gains
