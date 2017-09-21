import os
import cv2
import numpy as np
import random
from moviepy.editor import *
import mxnet as mx
from detect.detector import Detector

class video_generator:
    def __init__(self,video_path,fps,output_path='./result.mp4'):
        self.clip = VideoFileClip(video_path)
        self.output_path = output_path
        self.fps  = fps
        self.record = None
    def set_record(self,record):
        self.record = record
    def commit(self):
        def draw(img,bboxes):
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for b in bboxes:
                xmin,ymin,xmax,ymax = b[:]
                cv2.rectangle(img, (xmin,ymin),  (xmax,ymax),(255,255,0) ,thickness=2)
            return img
        def make_frame(t):
            idx = t*(self.clip.fps/self.fps)
            frm = self.clip.get_frame(t)
            height ,width = frm.shape[:2]
            for t,bboxes in self.record:
                if t==idx:        
                    frm = draw(frm,bboxes)
                else:
                    pass
            return frm
        new_clip = VideoClip(make_frame, duration=self.clip.duration) # 3-second clip
        new_clip.fps=self.clip.fps
        new_clip.to_videofile(self.output_path)



def get_mxnet_detector(net, prefix, epoch, data_shape, mean_pixels, ctx,batch_size = 1):
    detector = Detector(net, prefix, epoch, data_shape, mean_pixels, ctx=ctx,batch_size = 1)
    return detector


def img_preprocessing(img,data_shape):
    img = cv2.resize(img,(data_shape,data_shape))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2) 
    # img = img[np.newaxis, :] 
    return [mx.nd.array([img])]

def get_bboxes(img,dets,thresh = 0.5 ):
    
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    bboxes = []
    for i in range(dets.shape[0]):
        cls_id = int(dets[i, 0])
        if cls_id >= 0:
            score = dets[i, 1]
            if score > thresh:
                xmin = int(dets[i, 2] * width)
                ymin = int(dets[i, 3] * height)
                xmax = int(dets[i, 4] * width)
                ymax = int(dets[i, 5] * height)
                bboxes.append([xmin,ymin,xmax,ymax])
                # cv2.rectangle(img, (xmin,ymin),  (xmax,ymax),(255,255,0) ,thickness=2)
    # cv2.imwrite('./img.jpg',img)
    return bboxes
def main():
    #args
    net = None
    # prefix = os.path.join(os.getcwd(), 'model', 'yolo2_darknet19_416')
    # epoch = 240
    prefix = os.path.join(os.getcwd(), 'model', 'resnet50_yolov2_resnet50_416')
    epoch = 158
    
    data_shape = 416
    mean_pixels = (123,117,104)
    ctx = mx.gpu(0)

    detector = get_mxnet_detector(net, prefix, epoch, data_shape, mean_pixels, ctx=ctx,batch_size = 1)

    video_path = '/home/share/test_video/a1004s101_ch0.mp4'
    clip = VideoFileClip(video_path)
    record = []
    frames = clip.iter_frames(fps=clip.fps ,with_times = True)
    for t,frm in frames:
        data = img_preprocessing(frm,data_shape)
        det_batch = mx.io.DataBatch(data,[])
        detector.mod.forward(det_batch, is_train=False)
        detections = detector.mod.get_outputs()[0].asnumpy()
        result = []
        for i in range(detections.shape[0]):
            det = detections[i, :, :]
            res = det[np.where(det[:, 0] >= 0)[0]]
            result.append(res)
        bboxes = get_bboxes(frm,res)
        record.append([t,bboxes])

    vg = video_generator(video_path,fps = clip.fps)
    vg.set_record(record)
    vg.commit()

main()
