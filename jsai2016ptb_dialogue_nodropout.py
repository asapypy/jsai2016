#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import print_function
import argparse
import math
import sys
import time
from datetime import datetime

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import jsai2016models


parser = argparse.ArgumentParser()
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=39, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=650, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=500,
                    help='learning minibatch size')
parser.add_argument('--bproplen', '-l', type=int, default=35,
                    help='length of truncated BPTT')
parser.add_argument('--gradclip', '-c', type=int, default=1,
                    help='gradient norm threshold to clip')
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)

args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = args.epoch        # number of epochs
n_units = args.unit         # number of units per layer
batchsize = args.batchsize  # minibatch size
bprop_len = args.bproplen   # length of truncated BPTT
grad_clip = args.gradclip   # gradient norm threshold to clip

# Prepare dataset (preliminary download dataset by ./download.py)
vocab = {}


def load_data(filename):
    global vocab, n_vocab, rdict
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    words.append('<pad>')
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    rdict = dict((v,k) for k,v in vocab.iteritems())
    return dataset

# train_data = load_data('ptb.train.txt')
train_data = load_data('jsai2016ptb.train.txt')
print('train_data has ',len(train_data), ' words in this corpus.')
if args.test:
    train_data = train_data[:100]
# valid_data = load_data('ptb.valid.txt')
valid_data = load_data('jsai2016ptb.valid.txt')
print('valid_data has ',len(valid_data), ' words in this corpus.')
if args.test:
    valid_data = valid_data[:100]
# test_data = load_data('ptb.test.txt')
test_data = load_data('jsai2016ptb.test.txt')
print('test_data has ',len(test_data), ' words in this corpus.')
if args.test:
    test_data = test_data[:100]

print('#vocab =', len(vocab))

# Prepare RNNLM model, defined in net.py
# lm = jsai2016net.RNNLM(len(vocab), n_units)
lm = jsai2016models.JSAI2016DIALOGUE_NODROPOUT(len(vocab), n_units)
model = L.Classifier(lm)
model.compute_accuracy = True  # we want the accuracy
for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.SGD(lr=1.)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)


def evaluate(dataset):
    # Evaluation routine
    evaluator = model.copy()  # to use different state
    evaluator.predictor.reset_state()  # initialize state

    sum_log_perp = 0
    for i in six.moves.range(dataset.size - 1):
        x = chainer.Variable(xp.asarray(dataset[i:i + 1]), volatile='on')
        t = chainer.Variable(xp.asarray(dataset[i + 1:i + 2]), volatile='on')
        xQ = chainer.Variable(xp.asarray(dataset[i:i + 1]), volatile='on')
        tQ = chainer.Variable(xp.asarray(dataset[i + 1:i + 2]), volatile='on')
        xA = chainer.Variable(xp.asarray(dataset[i:i + 1]), volatile='on')
        tA = chainer.Variable(xp.asarray(dataset[i + 1:i + 2]), volatile='on')

        cnt_n = vocab['<cntnxt>']
        pad_n = vocab['<pad>']
        eos_n = vocab['<eos>']
        pos = 0;
        q_flg = True
        for j in xrange(len(x.data)):
            if x.data[j] == cnt_n and q_flg == True:
                xA.data[pos+1:j] = pad_n
                tA.data[pos+1:j] = pad_n
                pos = j
                q_flg = False
            if x.data[j] == eos_n and q_flg == False:
                xQ.data[pos+1:j] = pad_n
                tQ.data[pos+1:j] = pad_n
                pos = j
                q_flg = True

        if q_flg == False:
            xQ.data[pos+1:] = pad_n
            tQ.data[pos+1:] = pad_n
        else:
            xA.data[pos+1:] = pad_n
            tA.data[pos+1:] = pad_n
        xA.data[0] = pad_n
        tA.data[0] = pad_n

        loss = evaluator(x, t)
        loss += evaluator(xQ, tQ)
        loss += evaluator(xA, tA)
        sum_log_perp += loss.data
    return math.exp(float(sum_log_perp) / (dataset.size - 1))


# Learning loop
whole_len = train_data.shape[0]
print('whole_len=',whole_len)
# print(train_data.shape)
jump = whole_len // batchsize
print('batchsize=',batchsize,'  # length of minibatch')
print('jump=',jump,'  # number of minibatches')
cur_log_perp = xp.zeros(())

epoch = 0
start_at = time.time()
cur_at = start_at
accum_loss = 0
batch_idxs = list(range(batchsize))
print('going to train {} iterations'.format(jump * n_epoch))

for i in six.moves.range(jump * n_epoch):
    x = chainer.Variable(xp.asarray(
        [train_data[(jump * i + j) % whole_len] for j in batch_idxs]))
    t = chainer.Variable(xp.asarray(
        [train_data[(jump * i + j + 1) % whole_len] for j in batch_idxs]))
    xQ = chainer.Variable(xp.asarray(
        [train_data[(jump * i + j) % whole_len] for j in batch_idxs]))
    tQ = chainer.Variable(xp.asarray(
        [train_data[(jump * i + j + 1) % whole_len] for j in batch_idxs]))
    xA = chainer.Variable(xp.asarray(
        [train_data[(jump * i + j) % whole_len] for j in batch_idxs]))
    tA = chainer.Variable(xp.asarray(
        [train_data[(jump * i + j + 1) % whole_len] for j in batch_idxs]))

    cnt_n = vocab['<cntnxt>']
    pad_n = vocab['<pad>']
    eos_n = vocab['<eos>']
    pos = 0;
    q_flg = True
    for j in xrange(len(x.data)):
        if x.data[j] == cnt_n and q_flg == True:
            xA.data[pos+1:j] = pad_n
            tA.data[pos+1:j] = pad_n
            pos = j
            q_flg = False
        if x.data[j] == eos_n and q_flg == False:
            xQ.data[pos+1:j] = pad_n
            tQ.data[pos+1:j] = pad_n
            pos = j
            q_flg = True

    if q_flg == False:
        xQ.data[pos+1:] = pad_n
        tQ.data[pos+1:] = pad_n
    else:
        xA.data[pos+1:] = pad_n
        tA.data[pos+1:] = pad_n
    xA.data[0] = pad_n
    tA.data[0] = pad_n
#    print('x.data=', x.data)
#    print('t.data=', t.data)
#    print('xQ.data=', xQ.data)
#    print('tQ.data=', tQ.data)
#    print('xA.data=', xA.data)
#    print('tA.data=', tA.data)
#    print('cnt_n=', cnt_n)
#    print('eos_n=', eos_n)
#    print('pad_n=', pad_n)
#    print('x.data=', x.data[:10])
#   print('t.data=', t.data[:10])
#    sys.exit()
    loss_i = model(x, t)
    accum_loss += loss_i
    loss_i = model(xQ, tQ)
    accum_loss += loss_i
    loss_i = model(xA, tA)
    accum_loss += loss_i

    if (i + 1) % bprop_len == 0:  # Run truncated BPTT
        model.zerograds()
        accum_loss.backward()
        accum_loss.unchain_backward()  # truncate
        accum_loss = 0
        optimizer.update()

    if (i + 1) % 10000 == 0:
        now = time.time()
        throuput = 10000. / (now - cur_at)
        perp = math.exp(float(cur_log_perp) / 10000)
        print('iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(
            i + 1, perp, throuput))
        cur_at = now
        cur_log_perp.fill(0)

    if (i + 1) % jump == 0:
        epoch += 1
        print('evaluate')
        now = time.time()
        perp = evaluate(valid_data)
        print('epoch {} validation perplexity: {:.2f}'.format(epoch, perp))
        cur_at += time.time() - now  # skip time of evaluation

        # Save the model and the optimizer
        print('save the model')
        strtime = datetime.now().strftime('%Y%m%d%H%M%S')
        serializers.save_npz('jsai2016ptb_dialogue_nodropout_%s.model' % (strtime),
                             model)
        print('save the optimizer')
        serializers.save_npz('jsai2016ptb_dialogue_nodropout_%s.state' % (strtime),
                             optimizer)

        if epoch >= 6:
            optimizer.lr /= 1.2
            print('learning rate =', optimizer.lr)

    sys.stdout.flush()

# Evaluate on test dataset
print('test')
test_perp = evaluate(test_data)
print('test perplexity:', test_perp)

# Save the model and the optimizer
print('save the model')
# serializers.save_npz('rnnlm.model', model)
strtime = datetime.now().strftime('%Y%m%d%H%M%S')
serializers.save_npz('jsai2016ptb_dialogue_nodropout_%s.model' % (strtime), model)
print('save the optimizer')
# serializers.save_npz('rnnlm.state', optimizer)
serializers.save_npz('jsai2016ptb_dialogue_nodropout_%s.state' % (strtime), optimizer)
