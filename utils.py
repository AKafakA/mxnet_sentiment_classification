from math import exp
from math import log
from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
from mxnet import ndarray as nd
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib as mpl

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

def try_all_gpus():
    """Return all available GPUs, or [mx.gpu()] if there is no GPU"""
    ctx_list = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctx_list.append(ctx)
    except:
        pass
    if not ctx_list:
        ctx_list = [mx.cpu()]
    return ctx_list

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])

def evaluate_rnn_accuracy(data_iterator, model, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size)
            acc += nd.sum(model(X).argmax(axis=1)==y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read() # don't push too many operators into backend
    return acc.asscalar() / n


def grad_clipping(params, clipping_norm, ctx):
    """Gradient clipping."""
    if clipping_norm is not None:
        norm = nd.array([0.0], ctx)
        for p in params:
            norm += nd.sum(p.grad ** 2)
        norm = nd.sqrt(norm).asscalar()
        if norm > clipping_norm:
            for p in params:
                p.grad[:] *= clipping_norm / norm


def predict_rnn(rnn, prefix, num_chars, params, hidden_dim, ctx, idx_to_char,
                char_to_idx, get_inputs, is_lstm=False):
    """Predict the next chars given the prefix."""
    prefix = prefix.lower()
    state_h = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    if is_lstm:
        state_c = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for i in range(num_chars + len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        if is_lstm:
            Y, state_h, state_c = rnn(get_inputs(X), state_h, state_c, *params)
        else:
            Y, state_h = rnn(get_inputs(X), state_h, *params)
        if i < len(prefix)-1:
            next_input = char_to_idx[prefix[i+1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        output.append(next_input)
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn(rnn, is_random_iter, epochs, num_steps, hidden_dim, 
                          learning_rate, clipping_norm, batch_size,
                          pred_period, pred_len, seqs, get_params, get_inputs,
                          ctx, corpus_indices, idx_to_char, char_to_idx,
                          is_lstm=False):
    """Train an RNN model and predict the next item in the sequence."""
    if is_random_iter:
        data_iter = data_iter_random
    else:
        data_iter = data_iter_consecutive
    params = get_params()
    
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    for e in range(1, epochs + 1): 
        # If consecutive sampling is used, in the same epoch, the hidden state
        # is initialized only at the beginning of the epoch.
        if not is_random_iter:
            state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            if is_lstm:
                state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
        train_loss, num_examples = 0, 0
        for data, label in data_iter(corpus_indices, batch_size, num_steps, 
                                     ctx):
            # If random sampling is used, the hidden state has to be
            # initialized for each mini-batch.
            if is_random_iter:
                state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
                if is_lstm:
                    state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            with autograd.record():
                # outputs shape: (batch_size, vocab_size)
                if is_lstm:
                    outputs, state_h, state_c = rnn(get_inputs(data), state_h,
                                                    state_c, *params) 
                else:
                    outputs, state_h = rnn(get_inputs(data), state_h, *params)
                # Let t_ib_j be the j-th element of the mini-batch at time i.
                # label shape: (batch_size * num_steps)
                # label = [t_0b_0, t_0b_1, ..., t_1b_0, t_1b_1, ..., ].
                label = label.T.reshape((-1,))
                # Concatenate outputs:
                # shape: (batch_size * num_steps, vocab_size).
                outputs = nd.concat(*outputs, dim=0)
                # Now outputs and label are aligned.
                loss = softmax_cross_entropy(outputs, label)
            loss.backward()

            grad_clipping(params, clipping_norm, ctx)
            SGD(params, learning_rate)

            train_loss += nd.sum(loss).asscalar()
            num_examples += loss.size

        if e % pred_period == 0:
            print("Epoch %d. Training perplexity %f" % (e, 
                                               exp(train_loss/num_examples)))
            for seq in seqs:
                print(' - ', predict_rnn(rnn, seq, pred_len, params,
                      hidden_dim, ctx, idx_to_char, char_to_idx, get_inputs,
                      is_lstm))
            print()

def set_fig_size(mpl, figsize=(3.5, 2.5)):
    """set the size of the figure"""
    mpl.rcParams['figure.figsize'] = figsize


def evaluate_rnn(loss_func, data_iterator, model, hidden, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    total_loss = 0.0
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            out = model(X, hidden)
            
            acc += nd.sum(out.argmax(axis=1)==y).copyto(mx.cpu())
            cur_loss = loss_func(out, y).copyto(mx.cpu())
            total_loss += nd.mean(cur_loss).asscalar()
            
            n += y.size
        acc.wait_to_read() # don't push too many operators into backend
    return acc.asscalar() / n, total_loss / n

