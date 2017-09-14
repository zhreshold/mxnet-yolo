import os
import cv2
import numpy as np
import sys 
import mxnet as mx
import importlib
from timeit import default_timer as timer
from detect.detector import Detector


CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

img = './data/demo/dog.jpg'
net = 'darknet19_yolo'
sys.path.append(os.path.join(os.getcwd(), 'symbol'))
net = importlib.import_module("symbol_" + net) \
            .get_symbol(len(CLASSES), nms_thresh = 0.5, force_nms = True)
prefix = os.path.join(os.getcwd(), 'model', 'yolo2_darknet19_416')
epoch = 0
data_shape = 608
mean_pixels = (123,117,104)
ctx = mx.gpu(0)
batch = 3



detector = Detector(net, prefix, epoch, data_shape, mean_pixels, ctx=ctx,batch_size = batch)

ims = [cv2.resize(cv2.imread(img),(data_shape,data_shape)) for i in range(batch)]

def get_batch(imgs):
    img_len = len(imgs)
    l = []
    for i in range(batch):
        if i < img_len:
            img = np.swapaxes(imgs[i], 0, 2)
            img = np.swapaxes(img, 1, 2) 
            img = img[np.newaxis, :] 
            l.append(img[0])
        else:
            l.append(np.zeros(shape=(3, data_shape, data_shape)))
    l = np.array(l)
    return [mx.nd.array(l)]


data  = get_batch(ims)


start = timer()

for i in range(200):
    det_batch = mx.io.DataBatch(data,[])
    detector.mod.forward(det_batch, is_train=False)
    detections = detector.mod.get_outputs()[0].asnumpy()
    result = []
    for i in range(detections.shape[0]):
        det = detections[i, :, :]
        res = det[np.where(det[:, 0] >= 0)[0]]
        result.append(res)


time_elapsed = timer() - start
print("Detection time for {} images: {:.4f} sec , fps : {:.4f}".format(batch*200, time_elapsed , (batch*200/time_elapsed)))

print result

