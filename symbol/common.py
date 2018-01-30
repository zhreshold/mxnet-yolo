import mxnet as mx
import numpy as np

def conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", use_batchnorm=False):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="conv{}".format(name))
    if use_batchnorm:
        conv = mx.symbol.BatchNorm(data=conv, name="bn{}".format(name))
    relu = mx.symbol.Activation(data=conv, act_type=act_type, \
        name="{}{}".format(act_type, name))
    return relu

def stack_neighbor(from_layer, factor=2):
    """Downsample spatial dimentions and collapse to channel dimention by factor"""
    out = mx.sym.reshape(from_layer, shape=(0, 0, -4, -1, factor, -2))  # (b, c, h/2, 2, w)
    out = mx.sym.transpose(out, axes=(0, 1, 3, 2, 4))  # (b, c, 2, h/2, w)
    out = mx.sym.reshape(out, shape=(0, -3, -1, -2))  # (b, c * 2, h/2, w)
    out = mx.sym.reshape(out, shape=(0, 0, 0, -4, -1, factor))  # (b, c * 2, h/2, w/2, 2)
    out = mx.sym.transpose(out, axes=(0, 1, 4, 2, 3))  # (b, c*2, 2, h/2, w/2)
    out = mx.sym.reshape(out, shape=(0, -3, -1, -2))  # (b, c*4, h/2, w/2)
    return out
