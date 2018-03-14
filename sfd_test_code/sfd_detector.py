#!/usr/bin/env python2.7
from os.path import abspath, dirname 
import numpy as np

import sys
sys.path.insert(0, '../../python')
import caffe


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

    def detect(self, imgs, is_wider=False):
        """
        Detect elements on inputs.

        Parameters
        ----------
        inputs : iterable of (H x W x K) input ndarrays.
        is_wider : boolean, True if the image comes from WIDER FACES dataset

        Returns
        -------
        detections: list of 4D ndarrays of detections containing confidence, xmin, ymin, xmax and ymax 
        """
        # Scale to standardize input dimensions.
        detections = []
        # TODO: just do this for a single image, no need for batch processing
        for ix, in_ in enumerate(imgs):
            self.blobs[self.inputs[0]].reshape(1, 3, in_.shape[0], in_.shape[1])
            transformer = self.get_transformer(self.blobs[self.inputs[0]].data.shape)
            transformed_image = transformer.preprocess(self.inputs[0], in_)
            self.blobs[self.inputs[0]].data[...] = transformed_image
            detections.append(self.forward()['detection_out'])

        return detections

    def process_detections(self, detections, width, height, shrink=1):
        results = []
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        keep_index = np.where(det_conf >= 0)[0]
        det_conf = det_conf[keep_index]
        det_xmin = det_xmin[keep_index]
        det_ymin = det_ymin[keep_index]
        det_xmax = det_xmax[keep_index]
        det_ymax = det_ymax[keep_index]

        for i in range(det_conf.shape[0]):
            xmin = det_xmin[i] * width / shrink
            ymin = det_ymin[i] * height / shrink
            xmax = det_xmax[i] * width / shrink
            ymax = det_ymax[i] * height / shrink
            score = det_conf[i]
            results.append((score, xmin, ymin, xmax, ymax))

        return results
