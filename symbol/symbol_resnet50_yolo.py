import mxnet as mx
import resnet

def conv_act_layer(from_layer, name, num_filter, kernel=(3, 3), pad=(1, 1), \
    stride=(1,1), act_type="relu", use_batchnorm=True):
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
        stride=stride, num_filter=num_filter, name="{}".format(name))
    if use_batchnorm:
        conv = mx.symbol.BatchNorm(data=conv, name="bn_{}".format(name))
    if act_type in ['elu', 'leaky', 'prelu', 'rrelu']:
        relu = mx.symbol.LeakyReLU(data=conv, act_type=act_type,
        name="{}_{}".format(act_type, name), slope=0.1)
    elif act_type in ['relu', 'sigmoid', 'softrelu', 'tanh']:
        relu = mx.symbol.Activation(data=conv, act_type=act_type, \
        name="{}_{}".format(act_type, name))
    else:
        assert isinstance(act_type, str)
        raise ValueError("Invalid activation type: " + str(act_type))
    return relu

def get_symbol(num_classes=20, nms_thresh=0.5, force_nms=False, **kwargs):
    body = resnet.get_symbol(num_classes, 50, '3,224,224')
    conv1 = body.get_internals()['_plus12_output']
    conv2 = body.get_internals()['_plus15_output']
    # anchors
    anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
    num_anchor = len(anchors) // 2

    # extra layers
    conv7_1 = conv_act_layer(conv2, 'conv7_1', 1024, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')
    conv7_2 = conv_act_layer(conv7_1, 'conv7_2', 1024, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')

    # re-organize
    conv5_6 = mx.sym.stack_neighbor(data=conv1, kernel=(2, 2), name='stack_downsample')
    concat = mx.sym.Concat(*[conv5_6, conv7_2], dim=1)
    # concat = conv7_2
    conv8_1 = conv_act_layer(concat, 'conv8_1', 1024, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')
    pred = mx.symbol.Convolution(data=conv8_1, name='conv_pred', kernel=(1, 1),
        num_filter=num_anchor * (num_classes + 4 + 1))

    out = mx.contrib.symbol.YoloOutput(data=pred, num_class=num_classes,
        num_anchor=num_anchor, object_grad_scale=5.0, background_grad_scale=1.0,
        coord_grad_scale=1.0, class_grad_scale=1.0, anchors=anchors,
        nms_topk=400, warmup_samples=12800, name='yolo_output')
    return out
