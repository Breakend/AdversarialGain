# AdversarialGain

Download test datasets: https://github.com/harvardnlp/sent-summary

```
git submodule init
```

Run through the InferSent setup, downloading the Glove embeddings and the pretrained encoder.

Run the bootstrapped Gain analysis on the test data:

```
python infersent.py ./sumdata/Giga/input.txt ./sumdata/Giga/task1_ref0.txt
```
