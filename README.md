# KepKreator

## Team members:
Aszódi Zsombor XJ43M0 \
Babos Dávid Q1CGY7 \
Kovács Gergely JWV9WR

## The project:
The plan is to generate images using Conditional Generative Adversial Nets (cGANs). As other cGANs, our network will consist two models, a Generator and a Discriminator. The input to the Generator will be a noise vector and the output will be a generated image. The input of the Discriminator will be an image, either a 'real' image from the training dataset or a 'fake' image, created by the Generator. The Discriminator's role is to decide whether the input image is real or fake. We want to create a Conditional GAN, hence we will concatenate the input (both the Generator's and the Discriminator's) with a label. The goal is to generate hand-written numbers. At first we'll try creating one-digit numbers, than upgrade the model to create multi-digit ones. We will use the MNIST and the EMNIST datasets, to train our model. At the end we want to create an interface, where the user can give an input number, which the model than generates.

## Files in reposetory:
 - MNIST_preprocess.ipynb  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -> Imports the MNIST database
 - EMNIST_preprocess.ipynb  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -> Imports the EMNIST database

## Citations:
@article{lecun-mnisthandwrittendigit-2010,
  added-at = {2010-06-28T21:16:30.000+0200},
  author = {LeCun, Yann and Cortes, Corinna},
  biburl = {https://www.bibsonomy.org/bibtex/2935bad99fa1f65e03c25b315aa3c1032/mhwombat},
  groups = {public},
  howpublished = {http://yann.lecun.com/exdb/mnist/},
  interhash = {21b9d0558bd66279df9452562df6e6f3},
  intrahash = {935bad99fa1f65e03c25b315aa3c1032},
  keywords = {MSc _checked character_recognition mnist network neural},
  lastchecked = {2016-01-14 14:24:11},
  timestamp = {2016-07-12T19:25:30.000+0200},
  title = {{MNIST} handwritten digit database},
  url = {http://yann.lecun.com/exdb/mnist/},
  username = {mhwombat},
  year = 2010
}



