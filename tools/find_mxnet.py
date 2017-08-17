try:
    import mxnet0 as mx
except ImportError:
    import os, sys
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(curr_path, "../mxnet/python"))
    import mxnet as mx
