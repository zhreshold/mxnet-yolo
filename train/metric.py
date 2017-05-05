import mxnet as mx
import numpy as np


class MultiBoxMetric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, thresh=0.5, eps=1e-8):
        super(MultiBoxMetric, self).__init__(['Recall', 'IOU', 'BG'], 3)
        self.eps = eps
        self.thresh = thresh

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # temp = preds[1].asnumpy()
        # print np.reshape(temp, (-1, 32))[:3, :]
        # num_batch = preds[1].shape[0]
        # metric = mx.nd.slice_axis(preds[1].reshape((-1, num_batch)),
        #     axis=0, begin=0, end=3);
        # s = mx.nd.sum(metric, axis=1)
        # self.sum_metric[0] += s[0].asscalar()
        # self.num_inst[0] += 1
        # self.sum_metric[1] += s[1].asscalar()
        # self.num_inst[1] += 1
        # self.sum_metric[2] += s[2].asscalar()
        # self.num_inst[2] += 1
        # # assert(0 == 1)
        # # print "-----------------------"
        # # print preds[0].asnumpy()[0, :10, :]
        # # print preds[1].asnumpy()[0, :10, :]
        # return

        def calc_ious(b1, b2):
            assert b1.shape[1] == 4
            assert b2.shape[1] == 4
            num1 = b1.shape[0]
            num2 = b2.shape[0]
            b1 = np.repeat(b1.reshape(num1, 1, 4), num2, axis=1)
            b2 = np.repeat(b2.reshape(1, num2, 4), num1, axis=0)
            dw = np.maximum(0, np.minimum(b1[:, :, 2], b2[:, :, 2]) - \
                np.maximum(b1[:, :, 0], b2[:, :, 0]))
            dh = np.maximum(0, np.minimum(b1[:, :, 3], b2[:, :, 3]) - \
                np.maximum(b1[:, :, 1], b2[:, :, 1]))
            inter_area = dw * dh
            area1 = np.maximum(0, b1[:, :, 2] - b1[:, :, 0]) * \
                np.maximum(0, b1[:, :, 3] - b1[:, :, 1])
            area2 = np.maximum(0, b2[:, :, 2] - b2[:, :, 0]) * \
                np.maximum(0, b2[:, :, 3] - b2[:, :, 1])
            union_area = area1 + area2 - inter_area
            ious = inter_area / (union_area + self.eps)
            return ious

        def draw(boxes):
            import cv2
            w = 800
            h = 800
            canvas = np.ones((h, w, 3)) * 255
            for i in range(boxes.shape[0]):
                box = boxes[i, :]
                pt1 = (int(box[1] * h), int(box[0] * w))
                pt2 = (int(box[3] * h), int(box[2] * w))
                colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 255], [255, 255, 0]])
                color = colors[i % 5, :].astype(int)
                cv2.rectangle(canvas, pt1, pt2, color, 2)
            cv2.imshow('image', canvas)
            cv2.waitKey(0)

        out = preds[0].asnumpy()
        # draw(out[0, :, 2:])
        label = labels[0].asnumpy()
        for i in range(out.shape[0]):
            valid_mask = np.where(label[i, :, 0] >= 0)[0]
            valid_label = label[i, valid_mask, 1:5]
            ious = calc_ious(out[i, :, 2:6], valid_label)
            max_iou = np.amax(ious, axis=0)
            self.sum_metric[0] += np.sum(max_iou > self.thresh)
            self.num_inst[0] += max_iou.size
            self.sum_metric[1] += np.sum(max_iou)
            self.num_inst[1] += max_iou.size
            # for j in range(valid_mask.size):
            #     gt_id = label[i, valid_mask[j], 0]
            #     correct = np.intersect1d(np.where(ious[:, j] > self.thresh)[0],
            #         np.where(out[i, :, 0] == gt_id)[0])
            #     if correct.size > 0:
            #         self.sum_metric[0] += 1.
            #         max_iou = np.amax(ious[correct, j])
            #         self.sum_metric[1] += max_iou
            #         self.num_inst[1] += 1
            #     self.num_inst[0] += 1
            bg_mask = np.where(np.amax(ious, axis=1) < self.eps)[0]
            self.sum_metric[2] += np.sum(out[i, bg_mask, 1])
            self.num_inst[2] += bg_mask.size

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)
