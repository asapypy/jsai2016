#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import six
import matplotlib as plt
import nltk

with open('all_mlquestions.txt') as f:
    allstr = f.read()

tokens = nltk.word_tokenize(allstr.decode('utf-8'))
text = nltk.Text(tokens)
freq_dist = nltk.probability.FreqDist(text)
vocab = freq_dist.keys()
freq_dist.plot(50)
