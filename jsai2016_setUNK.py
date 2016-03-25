#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Shin Asakawa <asakawa@ieee.org>
# CreationData: 25/Mar/2016
# I, Shin Asakawa, have all rights reserved.

from __future__ import print_function
from __future__ import division
import six
import nltk
import matplotlib as plot

with open('all_mlquestions.txt') as f:
    allstr = f.read()

# tokens = nltk.word_tokenize(allstr.decode('utf-8'))
# tokens = nltk.word_tokenize(allstr.decode('utf-8').strip().lower())
tokens = nltk.word_tokenize(allstr.decode('utf-8').replace('-------------','').replace('Question=','').replace('comment no.= ','').replace('-->','').strip().lower())
text = nltk.Text(tokens)
freq_dist = nltk.probability.FreqDist(text)
vocab = freq_dist.keys()

print('# len(vocab)= ', len(vocab))
print('# len(text)=', len(text))
print('# len(frq_dist)= ', len(freq_dist))
print('# len(tokens)=', len(tokens))

# print(type(text))
# print(type(vocab))
freq_dist.plot(50)

# replace low frequence words with UNK tokens
mapping = nltk.defaultdict(lambda: 'UNK')

threshold_num = 5  # this varaible controls the value to be set
for v in list(vocab):
    if freq_dist[v] > threshold_num:
        mapping[v] = v
text2 = [mapping[v] for v in text]

for i in range(20):
    print(i, text[i], freq_dist[text[i]], text2[i])

cumulative = 0.0
for rank, word in enumerate(freq_dist):
    cumulative += freq_dist[word] * 100 / freq_dist.N()
    print('%3d %6.2f%% %s' % (rank+1, cumulative, word.encode('utf-8')))
    if cumulative > 25:
        break

text3 = set(text2)
print(len(text3))
# print(text3)
# print(text2)

# print(freq_dist.keys())
# print(freq_dist.values())
# print(list(mapping.keys()[:10]))
# print(len(text2), len(text))
# print(var(freq_dist.N))
# freq_dist['the']
# freq_dist.tabulate()[:10]
# help(nltk.probability.FreqDist)

# print(text2)
# for tok in text:
#     print(tok.encode('utf-8'), end=' ')
# print()
for tok in text2:
    print(tok.encode('utf-8'), end=' ')
