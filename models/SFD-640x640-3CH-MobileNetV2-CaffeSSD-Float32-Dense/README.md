
# SFD-640x640-3CH-MobileNetV2-CaffeSSD-Float32-Dense Model

__Input__
+ 640x640
+ 3 channels RGB

__Architecture__
+ MobileNet-V2
+ [MobileNetV2 Architecture - Pretrained Model COCO from tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
+ the tensorflow weights need to be transformed to caffe. Follw the instructions [here](https://github.com/chuanqi305/MobileNetv2-SSDLite) 
+ Features Layers : 'conv_13/expand', 'Conv_1', 'layer_19_2_2', 'layer_19_2_3', 'layer_19_2_4'


__Training__
+ Caffe SSD
+ Float32
+ Dense
+ Mean values: [104, 117, 123]

__Files__
+ [deploy](deploy.prototxt)
+ [train](train.prototxt)
+ [test](test.prototxt)
+ [trained model 30K iterations](https://drive.google.com/file/d/1ph0mumCcEKXtaupl_TYRc8Q7rWsEt_KD/view?usp=sharing)
+ [trained model 100K iterations](https://drive.google.com/file/d/1t3KQ-gZifDi68zi5Rjy2tYCbZm_-W8hw/view?usp=sharing)


__Results[30K iterations]__
+ Easy: %
+ Medium: %
+ hard: %

__Results[100K iterations]__
+ Easy: %
+ Medium: %
+ hard: %


### Notes

##### Training
+ MobileNetV2 model has too many differences from MobileNetV1, and they use different parameters for training.
+ The training of the model is very slow, and it takes too much memory. We used batch size of 2 images with iter_size=10, so that at the end the training is done with batch size = 20.
+ To solve the memory issue, these files nedd to be replaced in Caffe src/caffe/layers with thes [file](https://github.com/chuanqi305/MobileNetv2-SSDLite/tree/master/src). Don't forget to rebuild caffe again. 
+ To start training, the learning rate needs to be very small, otherwise the fluctuations in training loss and validation accuracy will be very big. I found that learning rate 0.00005 is the best depending on my observations for first 3k (validation accuracy is calculated each 1K).
+ During training, the behaviour of the training was normal until ~ 30K, then the fluctuations in validatidation accuracy curve started to increase, at the end the ccuracy was decreasing!! Next figure shows this behaviour.
![Alt text](SSD640x640-MobileNetV2-CaffeSSD-wider_training_loss.png)

+ To solve this problem, we can use the snapshot of the model at 30K iterations, to validate this solution we can also test the model at the end (100K iterations) to see the difference.

##### Testing
+ For testing, you should first modify the files in the src/ folder as explained in training notes. This should decrease the memory usage. After that, the testing should be done as usual.
+ I tested the model for WIDERA_FACE, but faced "out of memory" issue after event:25 num:3. Note that we use GPU with 16GB of RAM!
+ I tried to use the deploy model without batch_norm layer to decrease the memory usage, but it didn't work. 
+ To be able to test the model, the direct solution is to use GPU with more memory if possible. But also the testing codes could be modified so that it tests each crop of the image separately and free the memory after that.

