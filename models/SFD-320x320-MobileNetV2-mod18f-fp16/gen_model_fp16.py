import argparse
FLAGS = None

class Generator():

    def __init__(self):
      self.first_prior = True
      self.anchors = create_ssd_anchors()
      self.last = "data"

    def header(self, name):
      print("name: \"%s\"" % name)
    
    def data_deploy(self):
      print(
"""input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: %d
  dim: %d
}""" % (self.input_size, self.input_size))
    
    def data_train_classifier(self):
      print(
"""layer {
    name: "data"
    type: "Data"
    top: "data"
    top: "label"
    data_param{
        source: "%s"
        backend: LMDB
        batch_size: 64
    }
    transform_param {
        crop_size: %s
	mean_file: "imagenet.mean"
        mirror: true
    }
    include: { phase: TRAIN }
}
}""") 

    def data_train_ssd(self):
      print(
"""
  default_forward_type:  FLOAT16
  default_backward_type: FLOAT16
  default_forward_math:  FLOAT16
  default_backward_math: FLOAT16

  global_grad_scale: 32 #1000. #32

  layer {
  forward_type:  FLOAT
  backward_type: FLOAT

  name: "data"
  type: "AnnotatedData"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    mean_value: 104.0
    mean_value: 117.0
    mean_value: 123.0
    resize_param {
      prob: 1.0
      resize_mode: WARP
      height: %d
      width: %d
      interp_mode: LINEAR
      interp_mode: AREA
      interp_mode: NEAREST
      interp_mode: CUBIC
      interp_mode: LANCZOS4
    }
    emit_constraint {
      emit_type: CENTER
    }
    distort_param {
      brightness_prob: 0.5
      brightness_delta: 32.0
      contrast_prob: 0.5
      contrast_lower: 0.5
      contrast_upper: 1.5
      hue_prob: 0.5
      hue_delta: 18.0
      saturation_prob: 0.5
      saturation_lower: 0.5
      saturation_upper: 1.5
      random_order_prob: 0.0
    }
  }
  data_param {
    source: "%s"
    batch_size: 32
    backend: LMDB
  }
  annotated_data_param {
    batch_sampler {
      sampler {
        min_scale: 1.0
        max_scale: 1.0
        min_aspect_ratio: 1.0
        max_aspect_ratio: 1.0
      }
      sample_constraint {
        min_object_coverage: 1.0
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 1.0
        max_aspect_ratio: 1.0
      }
      sample_constraint {
        min_object_coverage: 1.0
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 1.0
        max_aspect_ratio: 1.0
      }
      sample_constraint {
        min_object_coverage: 1.0
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 1.0
        max_aspect_ratio: 1.0
      }
      sample_constraint {
        min_object_coverage: 1.0
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 1.0
        max_aspect_ratio: 1.0
      }
      sample_constraint {
        min_object_coverage: 1.0
      }
      max_sample: 1
      max_trials: 50
    }
    label_map_file: "%s"
  }
}"""  % (self.input_size, self.input_size, self.lmdb,  self.label_map))

    def data_test_ssd(self):
      print(
"""
  default_forward_type:  FLOAT16
  default_backward_type: FLOAT16
  default_forward_math:  FLOAT16
  default_backward_math: FLOAT16

  layer {
  forward_type:  FLOAT
  backward_type: FLOAT
  name: "data"
  type: "AnnotatedData"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mean_value: 104.0
    mean_value: 117.0
    mean_value: 123.0
    resize_param {
      prob: 1.0
      resize_mode: WARP
      height: %d
      width: %d
      interp_mode: LINEAR
    }
  }
  data_param {
    source: "%s"
    batch_size: 8
    backend: LMDB
  }
  annotated_data_param {
    batch_sampler {
    }
    label_map_file: "%s"
  }
}""" %  (self.input_size, self.input_size, self.lmdb,  self.label_map))


    def classifier_loss(self):
      print(
"""layer {
  name: "softmax"
  type: "Softmax"
  bottom: "%s"
  top: "prob"
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "prob"
  bottom: "label"
  top: "accuracy"
}
layer{
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "%s"
  bottom: "label"
  loss_weight: 1 
}""" % (self.last, self.last))

    def ssd_predict(self):
      print(
"""layer {
  name: "mbox_conf_reshape"
  type: "Reshape"
  bottom: "mbox_conf-_"
  top: "mbox_conf_reshape"
  reshape_param {
    shape {
      dim: 0
      dim: -1
      dim: %d
    }
  }
}
layer {
  name: "mbox_conf_sigmoid"
  type: "Sigmoid"
  bottom: "mbox_conf_reshape"
  top: "mbox_conf_sigmoid"
}
layer {
  name: "mbox_conf_flatten"
  type: "Flatten"
  bottom: "mbox_conf_sigmoid"
  top: "mbox_conf_flatten"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "detection_out"
  type: "DetectionOutput"
  bottom: "mbox_loc-_"
  bottom: "mbox_conf_flatten"
  bottom: "mbox_priorbox"
  top: "detection_out"
  include {
    phase: TEST
  }
  detection_output_param {
    num_classes: %d
    share_location: true
    background_label_id: 0
    nms_param {
      nms_threshold: 0.3
      top_k: 5000
    }
    code_type: CENTER_SIZE
    keep_top_k: 750
    confidence_threshold: 0.05
  }
}""" % (self.class_num, self.class_num))

    def ssd_test(self):
      print(
"""layer {
  name: "mbox_conf_reshape"
  type: "Reshape"
  bottom: "mbox_conf-_"
  top: "mbox_conf_reshape"
  reshape_param {
    shape {
      dim: 0
      dim: -1
      dim: %d
    }
  }
}
layer {
  name: "mbox_conf_sigmoid"
  type: "Sigmoid"
  bottom: "mbox_conf_reshape"
  top: "mbox_conf_sigmoid"
}
layer {
  name: "mbox_conf_flatten"
  type: "Flatten"
  bottom: "mbox_conf_sigmoid"
  top: "mbox_conf_flatten"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "detection_out"
  type: "DetectionOutput"
  bottom: "mbox_loc-_"
  bottom: "mbox_conf_flatten"
  bottom: "mbox_priorbox"
  top: "detection_out"
  include {
    phase: TEST
  }
  detection_output_param {
    num_classes: %d
    share_location: true
    background_label_id: 0
    nms_param {
      nms_threshold: 0.3
      top_k: 5000
    }
    code_type: CENTER_SIZE
    keep_top_k: 750
    confidence_threshold: 0.05
  }
}
layer {
  name: "detection_eval"
  type: "DetectionEvaluate"
  bottom: "detection_out"
  bottom: "label"
  top: "detection_eval"
  include {
    phase: TEST
  }
  detection_evaluate_param {
    num_classes: %d
    background_label_id: 0
    overlap_threshold: 0.5
    evaluate_difficult_gt: false
  }
}""" % (self.class_num, self.class_num, self.class_num))

    def ssd_loss(self):
      print(
"""layer {
  forward_type:  FLOAT
  backward_type: FLOAT
  name: "mbox_loss"
  type: "MultiBoxLoss"
  bottom: "mbox_loc-_"
  bottom: "mbox_conf-_"
  bottom: "mbox_priorbox"
  bottom: "label"
  top: "mbox_loss"
  include {
    phase: TRAIN
  }
  propagate_down: true
  propagate_down: true
  propagate_down: false
  propagate_down: false
  loss_param {
    normalization: VALID
  }
  multibox_loss_param {
    loc_loss_type: SMOOTH_L1
    conf_loss_type: LOGISTIC
    loc_weight: 1.0
    num_classes: %d
    share_location: true
    match_type: PER_PREDICTION
    overlap_threshold: 0.3
    use_prior_for_matching: true
    background_label_id: 0
    use_difficult_gt: true
    neg_pos_ratio: 3.0
    neg_overlap: 0.35
    code_type: CENTER_SIZE
    ignore_cross_boundary_bbox: false
    mining_type: MAX_NEGATIVE
  }
}""" % self.class_num)

    def concat_boxes(self, convs):
      for layer in ["loc-_", "conf-_"]:
        bottom =""
        for cnv in convs:
          bottom += "\n  bottom: \"%s_mbox_%s_flat\"" % (cnv, layer)
        print(
"""layer {
  name: "mbox_%s"
  type: "Concat"%s
  top: "mbox_%s"
  concat_param {
    axis: 1
  }
}""" % (layer, bottom, layer))

      bottom =""
      for cnv in convs:
        bottom += "\n  bottom: \"%s_mbox_priorbox\"" % cnv
      print(
"""layer {
  name: "mbox_priorbox"
  type: "Concat"%s
  top: "mbox_priorbox"
  concat_param {
    axis: 2
  }
}""" % bottom)

    def conv(self, name, out, kernel, stride=1, group=1, bias=False, bottom=None):

      if self.stage == "deploy" and self.nobn: #for deploy, merge bn to bias, so bias must be true
          bias = True

      if bottom is None:
          bottom = self.last
      padstr = ""
      if kernel > 1:
          padstr = "\n    pad: %d" % (kernel / 2)
      groupstr = ""
      if group > 1:
          groupstr = "\n    group: %d\n    #engine: CAFFE" % group
      stridestr = ""
      if stride > 1:
          stridestr = "\n    stride: %d" % stride 
      bias_lr_mult = ""
      bias_filler = ""
      if bias == True:
          bias_filler = """
    bias_filler {
      type: "constant"
      value: 0.0
    }"""
          bias_lr_mult = """
  param {
    lr_mult: 2.0
    decay_mult: 0.0
  }"""
      biasstr = ""
      if bias == False:
          biasstr = "\n    bias_term: false"
      print(
"""layer {
  name: "%s"
  type: "Convolution"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }%s
  convolution_param {
    num_output: %d%s%s
    kernel_size: %d%s%s
    weight_filler {
      type: "msra"
    }%s
  }
}""" % (name, bottom, name, bias_lr_mult, out, biasstr, padstr, kernel, stridestr, groupstr, bias_filler))
      self.last = name
    
    def bn(self, name):
      if self.stage == "deploy" and self.nobn:  #deploy does not need bn, you can use merge_bn.py to generate a new caffemodel
         return
      eps_str = ""
      if self.eps != 1e-5:
          eps_str = "\n  batch_norm_param {\n    eps: 0.001\n  }"
      print(
"""layer {
  name: "%s/bn"
  type: "BatchNorm"
  bottom: "%s"
  top: "%s"%s
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
}
layer {
  name: "%s/scale"
  type: "Scale"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1.0
    decay_mult: 0.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 0.0
  }
  scale_param {
    filler {
      value: 1
    }
    bias_term: true
    bias_filler {
      value: 0
    }
  }
}""" % (name,name,name,eps_str,name,name,name))
      self.last = name
    
    def relu(self, name):
      relu_str = "ReLU"
      if self.relu6:
         relu_str = "ReLU6"
      print(
"""layer {
  name: "%s/relu"
  type: "%s"
  bottom: "%s"
  top: "%s"
}""" % (name, relu_str, name, name))
      self.last
      self.last = name
    
    
    def conv_bn_relu(self, name, num, kernel, stride):
      self.conv(name, num, kernel, stride)
      self.bn(name)
      self.relu(name)

    def conv_bn_relu_with_factor(self, name, outp, kernel, stride):
      outp = int(outp * self.size)
      self.conv(name, outp, kernel, stride)
      self.bn(name)
      self.relu(name)

    def conv_ssd(self, name, stage, inp, outp):
      stage = str(stage)
      self.conv_expand(name + '_1_' + stage, inp, outp / 2)
      self.conv_depthwise(name + '_2_' + stage + '/depthwise', outp / 2, 2)
      self.conv_expand(name + '_2_' + stage, outp / 2, outp)

    def conv_block(self, name, inp, t, outp, stride, sc):
      last_block = self.last
      self.conv_expand(name + '/expand_', inp, t * inp)  ## name modified when change expand ratio
      self.conv_depthwise(name + '/depthwise_', t * inp, stride)
      if sc:
         self.conv_project(name + '/project_', t * inp, outp)
         self.shortcut(last_block, name)
      else:
         self.conv_project(name + '/project_', t * inp, outp)
    
    def conv_depthwise(self, name, inp, stride):
      inp = int(inp * self.size)
      self.conv(name, inp, 3, stride, inp)
      self.bn(name)
      self.relu(name)

    def conv_expand(self, name, inp, outp):
      inp = int(inp * self.size)
      outp = int(outp * self.size)
      self.conv(name, outp, 1)
      self.bn(name)
      self.relu(name)

    def conv_project(self, name, inp, outp):
      inp = int(inp * self.size)
      outp = int(outp * self.size)
      self.conv(name, outp, 1)
      self.bn(name)

    def conv_dw_pw(self, name, inp, outp, stride):
      inp = int(inp * self.size)
      outp = int(outp * self.size)
      name1 = name + "/depthwise"
      self.conv(name1, inp, 3, stride, inp)
      self.bn(name1)
      self.relu(name1)
      name2 = name 
      self.conv(name2, outp, 1)
      self.bn(name2)
      self.relu(name2)
    
    def shortcut(self, bottom, top):
      print(
"""layer {
  name: "%s/sum"
  type: "Eltwise"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
}""" % (top, bottom, self.last, top))
      self.last = top
    def ave_pool(self, name):
      print(
"""layer {
  name: "%s"
  type: "Pooling"
  bottom: "%s"
  top: "%s"
  pooling_param {
    pool: AVE
    global_pooling: true
  }
}""" % (name, self.last, name))
      self.last = name
    
    def permute(self, name):
      print(
"""layer {
  name: "%s_perm"
  type: "Permute"
  bottom: "%s"
  top: "%s_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}""" % (name, name, name))
      self.last = name + "_perm"
    
    def flatten(self, name):
      print(
"""layer {
  name: "%s_flat"
  type: "Flatten"
  bottom: "%s_perm"
  top: "%s_flat"
  flatten_param {
    axis: 1
  }
}""" % (name, name, name))
      self.last = name + "_flat"
    
    def mbox_prior(self, name, min_size, max_size, step, aspect_ratio):
      min_box = min_size
      #min_box = self.input_size * min_size
      max_box_str = ""
      aspect_ratio_str = ""
      min_box_str = ""
      step_str = ""
      if max_size is not None:
          max_box = self.input_size * max_size
          max_box_str = "\n    max_size: %.1f" % max_box
      for ar in aspect_ratio:
          aspect_ratio_str += "\n    aspect_ratio: %.1f" % ar
      for ms in min_size:
          min_box_str += "\n    min_size: %.1f" % ms
      for st in step:
          step_str += "\n    step: %.1f" % st
      if self.stage == "deploy":
          print(
"""layer {
  name: "%s_mbox_priorbox"
  type: "PriorBox"
  bottom: "%s"
  bottom: "data"
  top: "%s_mbox_priorbox"
  prior_box_param {%s%s%s
    flip: false
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2%s
    offset: 0.5
  }
}""" % (name, name, name, min_box_str, max_box_str, aspect_ratio_str, step_str))
      else:
          print(
"""layer {
  forward_type:  FLOAT
  backward_type: FLOAT
  name: "%s_mbox_priorbox"
  type: "PriorBox"
  bottom: "%s"
  bottom: "data"
  top: "%s_mbox_priorbox"
  prior_box_param {%s%s%s
    flip: false
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2%s
    offset: 0.5
  }
}""" % (name, name, name, min_box_str, max_box_str, aspect_ratio_str, step_str))



    def mbox_conf(self, bottom, num):
       name = bottom + "_mbox_conf-_"
       self.conv(name, num, 3, bias=True, bottom=bottom)
       self.permute(name)
       self.flatten(name)
    def mbox_loc(self, bottom, num):
       name = bottom + "_mbox_loc-_"
       self.conv(name, num, 3, bias=True, bottom=bottom)
       self.permute(name)
       self.flatten(name)

    def mbox(self, bottom, num):
       self.mbox_loc(bottom, num * 4)
       self.mbox_conf(bottom, num * self.class_num)
       min_size, step = self.anchors[0]
       if self.first_prior:
           self.mbox_prior(bottom, min_size, None, step, []) # [2.0])
           self.first_prior = False
       else:
           self.mbox_prior(bottom, min_size, None, step, []) #[2.0,3.0])
       self.anchors.pop(0)

    def fc(self, name, output):
      print(
"""layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  param { lr_mult: 1  decay_mult: 1 }
  param { lr_mult: 2  decay_mult: 0 }
  inner_product_param {
    num_output: %d
    weight_filler { type: "msra" }
    bias_filler { type: "constant"  value: 0 }
  }
}""" % (name, self.last, name, output))
      self.last = name
    
    def reshape(self, name, output):
      print(
"""layer {
    name: "%s"
    type: "Reshape"
    bottom: "%s"
    top: "%s"
    reshape_param { shape { dim: -1 dim: %s dim: 1 dim: 1 } }
}""" % ( name, self.last, name, output))
      self.last = name

    def generate(self, stage, gen_ssd, size, class_num, nobn, eps, relu6):
      self.class_num = class_num
      self.lmdb = FLAGS.lmdb
      if FLAGS.lmdb == "":
          if stage == "train":
              self.lmdb = "trainval_lmdb"
          elif stage == "test":
              self.lmdb = "test_lmdb"
      self.label_map = FLAGS.label_map
      self.stage = stage
      self.nobn = nobn
      self.eps = eps
      self.relu6 = relu6
      if gen_ssd:
          self.input_size = 320
      else:
          self.input_size = 224
      self.size = size
      self.class_num = class_num

      if gen_ssd:
          self.header("MobileNetv2-SSD")
      else:
          self.header("MobileNetv2")
      if stage == "train":
          if gen_ssd:
              assert(self.lmdb is not None)
              assert(self.label_map is not None)
              self.data_train_ssd()
          else:
              assert(self.lmdb is not None)
              self.data_train_classifier()
      elif stage == "test":
          self.data_test_ssd()
      else:
          self.data_deploy()
      t = 2
      t_ = 4
      self.conv_bn_relu_with_factor("Conv", 24, 3, 2)
      self.conv_bn_relu_with_factor("Conv_a", 24, 3, 1)
      self.conv_depthwise("conv/depthwise", 24, 1)
      self.conv_project("conv/project", 24, 24)
      self.conv_block("conv_1", 24,  t, 48 , 2, False) #
      self.conv_block("conv_2", 48,  t, 48, 1, True)
      self.conv_block("conv_3", 48,  t, 48, 1, True)  # 
      self.conv_block("conv_4", 48,  t, 48, 1, True)
      self.conv_block("conv_5", 48,  t, 48, 1, True) #
      self.conv_block("conv_6", 48,  t, 48, 1, True)
      self.conv_block("conv_7", 48,  t, 96, 2, False)
      self.conv_block("conv_8", 96,  t, 96, 1, True) #
      self.conv_block("conv_9", 96,  t, 96, 1, True)
      self.conv_block("conv_10", 96,  t, 96, 1, True) # 
      self.conv_block("conv_11", 96,  t, 96, 1, True)
      self.conv_block("conv_12", 96,  t, 96, 1, True)
      self.conv_block("conv_13", 96,  t, 96, 1, True)
      self.conv_block("conv_14", 96,  t, 128, 2, False)
      self.conv_block("conv_15", 128,  t, 128, 1, True)
      self.conv_block("conv_16", 128,  t, 128, 1, True)
      self.conv_block("conv_17", 128,  t, 128, 1, True)
      self.conv_block("conv_18", 128,  t, 128, 1, True)
      self.conv_bn_relu_with_factor("Conv_1", 160, 1, 1)
      if gen_ssd is True:
          self.conv_ssd("layer_19", 2, 160, 160)
          self.conv_ssd("layer_19", 3, 160, 160)
          self.conv_ssd("layer_19", 4, 160, 160)
          #self.conv_ssd("layer_19", 5, 256, 128)
          self.mbox("conv_14/expand_", 2)
          self.mbox("Conv_1", 1)
          self.mbox("layer_19_2_2", 1)
          self.mbox("layer_19_2_3", 1)
          self.mbox("layer_19_2_4", 1)
          #self.mbox("layer_19_2_5", 6)
          self.concat_boxes(['conv_14/expand_', 'Conv_1', 'layer_19_2_2', 'layer_19_2_3', 'layer_19_2_4']) 
          if stage == "train":
             self.ssd_loss()
          elif stage == "deploy":
             self.ssd_predict()
          else:
             self.ssd_test()
      else:
          self.ave_pool("pool")
          self.conv("fc", class_num, 1, 1, 1, True)
          if stage == "train":
             self.classifier_loss()

   
def create_ssd_anchors(num_layers=6,
                       min_scale=0.2,
                       max_scale=0.95):
  #box_specs_list = []
  #scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
  #          for i in range(num_layers)] + [1.0]
  #return zip(scales[:-1], scales[1:])
  scales = [[16, 32], [64], [128], [256], [512] ]
  steps =  [[8], [16], [32], [64], [128] ]
  #scales = [[64], [128], [256], [512] ]
  #steps =  [[16], [32], [64], [128] ]
  return zip(scales, steps)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-s','--stage',
      type=str,
      default='train',
      help='The stage of prototxt, train|test|deploy.'
  )
  parser.add_argument(
      '-d','--lmdb',
      type=str,
      default=None,
      help='The training or testing database'
  )
  parser.add_argument(
      '-l','--label-map',
      type=str,
      default="labelmap.prototxt",
      help='The label map for ssd training.'
  )
  parser.add_argument(
      '--classifier',
      action='store_true',
      help='Default generate ssd, if this is set, generate classifier prototxt.'
  )
  parser.add_argument(
      '--size',
      type=float,
      default=1.0,
      help='The size of mobilenet channels, support 1.0, 0.75, 0.5, 0.25.'
  )
  parser.add_argument(
      '-c', '--class-num',
      type=int,
      required=True,
      help='Output class number, include the \'backgroud\' class. e.g. 21 for voc.'
  )
  parser.add_argument(
      '--no_batchnorm',
      action='store_true',
      help='for deploy, generate a deploy.prototxt without batchnorm and scale'
  )
  parser.add_argument(
      '--eps',
      type=float,
      default=0.001,
      help='eps parameter of BatchNorm layers, default is 1e-5'
  )
  parser.add_argument(
      '--relu6',
      action='store_true',
      help='replace ReLU layers by ReLU6'
  )
  FLAGS, unparsed = parser.parse_known_args()
  gen = Generator()
  gen.generate(FLAGS.stage, not FLAGS.classifier, FLAGS.size, FLAGS.class_num, FLAGS.no_batchnorm, FLAGS.eps, FLAGS.relu6)
