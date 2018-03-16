#!/usr/bin/env python2.7
from os.path import abspath, dirname 
import cv2
import numpy as np

import sys
sys.path.insert(0, '../../python')
import caffe


# Default models if none is passed to the __init__ method, assuming the `sfd_test_code` is in ${CAFFE_ROOT}/SFD/sfd_test_code
# These models are automatically downloaded by SFD/scripts/data/download_model.sh
MODEL_DEF = '{}/../../models/VGGNet/WIDER_FACE/SFD_trained/deploy.prototxt'.format(abspath(dirname(__file__)))
MODEL_WEIGHTS = '{}/../../models/VGGNet/WIDER_FACE/SFD_trained/SFD.caffemodel'.format(abspath(dirname(__file__)))


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
        if type(device) is int:
            caffe.set_device(device)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        if model_file is None:
            model_file = MODEL_DEF
        if pretrained_file is None:
            pretrained_file = MODEL_WEIGHTS

        caffe.Net.__init__(self, model_file, caffe.TEST, weights=pretrained_file)

    def get_transformer(self, shape):
        in_ = self.inputs[0]
        transformer = caffe.io.Transformer({in_: shape})
        transformer.set_transpose(in_, (2, 0, 1))
        transformer.set_mean(in_, np.array([104, 117, 123]))
        transformer.set_raw_scale(in_, 255)
        transformer.set_channel_swap(in_, (2, 1, 0))
        return transformer

    def detect(self, img, shrink=(1, 1)):
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
        if shrink[0] != 1 or shrink[1] != 1:
            img = cv2.resize(img, None, None, fx=shrink[0], fy=shrink[1], interpolation=cv2.INTER_LINEAR)

        height = img.shape[0]
        width = img.shape[1]

        self.blobs[self.inputs[0]].reshape(1, 3, height, width)
        transformer = self.get_transformer(self.blobs[self.inputs[0]].data.shape)
        transformed_image = transformer.preprocess(self.inputs[0], img)
        self.blobs[self.inputs[0]].data[...] = transformed_image
        detections = self.forward()['detection_out']

        # Adjust SFD output to image size
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3] * width / shrink[0]
        det_ymin = detections[0, 0, :, 4] * height / shrink[1]
        det_xmax = detections[0, 0, :, 5] * width / shrink[0]
        det_ymax = detections[0, 0, :, 6] * height / shrink[1]
        det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

        keep_index = np.where(det[:, 4] >= 0)[0]
        det = det[keep_index, :]

        return det
