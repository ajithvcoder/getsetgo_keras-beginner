## Using pretrained Convolutional Neural Network models(CNN) in keras

### credits _/\\_
 > -https://mlblr.com/
 > -https://keras.io/
 > -https://colab.research.google.com/

codes used here are executed and tested only in colab.research.google.com 
please make sure you have enabled GPU access from notebook settings for faster execution process

### CNN models
Models used here are pretrained on ImageNet dataset  ImageNet Large Scale Visual Recognition Challenge (ILSVRC) with python and keras deep learning library.

1.Resnet50 model 
	-Classifying input image with probabilities [notebook link](https://github.com/ajithvallabai/getsetgo_keras-beginner\notebooks\Resnet50_imagenet_prediction.ipynb)
	-Visualizing resnet50 model [notebook link](https://github.com/ajithvallabai/getsetgo_keras-beginner\notebooks\Resnet50_visualization.ipynb)

2.VGG16 model
	-Classifying input image with probabilities [notebook link](https://github.com/ajithvallabai/getsetgo_keras-beginner\notebooks\Vgg16_imagenet_prediction.ipynb)
	-Visuvalizing vgg16 model [notebook link](https://github.com/ajithvallabai/getsetgo_keras-beginner\notebooks\Vgg16_visualization.ipynb)

3.VGG19 model
	-Classifying input image with probabilities  [notebook link](https://github.com/ajithvallabai/getsetgo_keras-beginner\notebooks\VGG19_imagenet_prediction.ipynb)
	-Extracting last layer of vgg19 and using them to predict  [notebook link](https://github.com/ajithvallabai/getsetgo_keras-beginner\notebooks\VGG19_imagenet_extractinglayer.ipynb)
	-Visuvalizing model extracted model [notebook link](https://github.com/ajithvallabai/getsetgo_keras-beginner\notebooks\VGG19_visualization.ipynb)

4.InceptionV3 model
	-Classifying input image with probabilities [notebook link](https://github.com/ajithvallabai/getsetgo_keras-beginner\notebooks\Inceptionv3_imagenet_prediction.ipynb)
	-Extracting last layer of vgg19 and using them to predict  [notebook link](https://github.com/ajithvallabai/getsetgo_keras-beginner\notebooks\Inceptionv3_extractinglayer.ipynb)
	-Visuvalizing model extracted model [notebook link](https://github.com/ajithvallabai/getsetgo_keras-beginner\notebooks\Inceptionv3_vizualizing.ipynb)

What is a pretrained model?
Models that has been already trained on a particular data set with number of classes

Use of pretrained model:
Instead of training from scrath/starting with random weight initialiation pre-trained
models can be used with other datasets.
[keywords: fine tuning-transfer learning]

### Model info:

Resnet50 model
Architecture speciality:Use of residual blocks(skip connections) enabled training much deeper 
network by handling vanishing and exploding gradient problems
https://arxiv.org/abs/1512.03385


VGG16 model
Architecture speciality: Use of 3x3 convolution filters with depth of 16 weight layers
https://arxiv.org/abs/1409.1556

VGG19 model
Architecture speciality: Use of 3x3 convolution filters with depth of 19 weight layers
https://arxiv.org/abs/1409.1556

Inceptionv3 model
Architecture speciality:1x1,3x3,5x5,max pooling convolutions performed with less computation cost-sparsely deep connected network-Hebbian principle(neurons that fire together wire 
together)
https://arxiv.org/pdf/1409.4842.pdf


