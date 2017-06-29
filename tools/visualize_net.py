from __future__ import print_function
import find_mxnet
import mxnet as mx
import importlib
import argparse
import sys

parser = argparse.ArgumentParser(description='network visualization')
parser.add_argument('--network', type=str, default='darknet19_yolo',
                    help = 'the cnn to use')
parser.add_argument('--num-classes', type=int, default=20,
                    help='the number of classes')
parser.add_argument('--data-shape', type=int, default=416,
                    help='set image\'s shape')
parser.add_argument('--train', action='store_true', default=False, help='show train net')
args = parser.parse_args()

sys.path.append('../symbol')

if not args.train:
    net = importlib.import_module("symbol_" + args.network).get_symbol(args.num_classes)
    a = mx.viz.plot_network(net, shape={"data":(1,3,args.data_shape,args.data_shape),  "yolo_output_label":(1, 10, 6)}, \
        node_attrs={"shape":'rect', "fixedsize":'false'})
    a.render("yolo2_" + args.network)
else:
    net = importlib.import_module("symbol_" + args.network).get_symbol_train(args.num_classes)
    print(net.tojson())
