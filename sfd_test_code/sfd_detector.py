#!/usr/bin/env python2.7
import cv2
import numpy as np

import sys
#sys.path.insert(0, '../../../caffe-jacinto/python')
#sys.path.insert(0, '../../../caffe-0.17/python')
sys.path.insert(0, '../../python')
import caffe


class SFD_NET(caffe.Net):
    """
    This class extends Net for SFD

    Parameters
    ----------
    model_file, pretrained_file: prototxt and caffemodel respectively. 
        If not provided, will use default ones assumming this script is in {$sfd_root}/sfd_test_code/

    mean, input_scale, raw_scale, channel_swap: params for
        preprocessing options.
    device, int if set, then tries to use the GPU with that device order
    """
    def __init__(self, model_file=None, pretrained_file=None,
                 mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None, device=None):
        if device >= 0:
            caffe.set_mode_gpu()
            caffe.set_device(device)
        else:
            caffe.set_mode_cpu()

        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

    def get_transformer(self, shape):
        _, channels, _, _ = shape
        in_ = self.inputs[0]
        transformer = caffe.io.Transformer({in_: shape})
        transformer.set_transpose(in_, (2, 0, 1))
        transformer.set_raw_scale(in_, 255)

        if channels == 1:
            transformer.set_mean(in_, np.array([np.mean([104, 117, 123])]))
        elif channels == 3:
            transformer.set_mean(in_, np.array([104, 117, 123]))
            transformer.set_channel_swap(in_, (2, 1, 0))
        else:
            raise Exception("{} channels images are not supported".format(c))

        return transformer

    def detect(self, img, shrink=1):
        """
        Detect elements on a single input image.

        Parameters
        ----------
        inputs : (H x W x K) ndarray.
        shrink: float, ratio to adjust output detections 

        Returns
        -------
        detections: np.array of detections containing xmin, ymin, xmax, ymax and confidence
        """
        if shrink != 1:
            _, _, c = img.shape
            img = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
            img = img.reshape(img.shape[0], img.shape[1], c)

        height = img.shape[0]
        width = img.shape[1]
        chann = img.shape[2]

        self.blobs[self.inputs[0]].reshape(1, chann, height, width)
        transformer = self.get_transformer(self.blobs[self.inputs[0]].data.shape)
        transformed_image = transformer.preprocess(self.inputs[0], img)
        self.blobs[self.inputs[0]].data[...] = transformed_image
        detections = self.forward()['detection_out']

        # Adjust SFD output to image size
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3] * width / shrink
        det_ymin = detections[0, 0, :, 4] * height / shrink
        det_xmax = detections[0, 0, :, 5] * width / shrink
        det_ymax = detections[0, 0, :, 6] * height / shrink
        det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

        keep_index = np.where(det[:, 4] >= 0)[0]
        det = det[keep_index, :]
        # Avoid bboxes that have overflowed
        avoid = list(set(np.where((det == np.inf) | (det == -np.inf))[0]))
        mask = np.ones(len(det), dtype=np.bool)
        mask[avoid] = 0
        det = det[mask, :]

        return det
