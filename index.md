## Bird Kaggle Competition for CSE 455

This website describes our process for testing different resnet architectures on the birds kaggle competition from [here](https://www.kaggle.com/c/birds21sp) for the CSE 455 course at University of Washington for Spring 2021.

Group: Akshit Arora, Braxton Kinney, Frederick Huyan, Leo Liao, Nikolai Scheel

## Simple Experiments

Before actually training models for the Kaggle competition, we started with some basic experimentation on MNIST and CIFAR to help us understand neural networks. If you would like to directly read what we did for the Kaggle competition, [skip to the "Problem Statement" section below](#Problem-Statement)

[Click here to view the video detailing these experiments.](https://drive.google.com/file/d/1ejyFTdBus_HcdmKOVBltC2XFb7Lr0fPX/view?usp=sharing)

[Click here to view the slideshow detailing this information.](https://docs.google.com/presentation/d/1N51GK0HlqaiIMy8HTi-qnWjPuc3_tfQDKLEJMtLP91U/edit?usp=sharing)

[Click here to view the raw experiment data in spreadsheet form.](https://docs.google.com/spreadsheets/d/1_hWArK0Wu42o0rZcaaQJxyeBj_8Es8VpY7yecR936ag/edit?usp=sharing)

## Problem Statement

We wanted to build a neural network that could classify 555 different species of birds. 

## Data

Our training data consisted of 38562 high resolution images split between the 555 different species of birds to identify, and 10000 images in the test dataset.  

## Methodology

To tackle this problem, we used transfer learning with ResNet using different ResNet architectures including ResNet18, ResNet50, and ResNet152 to see how each one compared to one another. We used the built in ResNet architecture in torchvision and hosted our model on Google Colab. 

We split off 1000 samples from the training set to use as a validation set to check for overfitting. 

As these models took hours to train, our approach was to start with a relatively large learning rate (0.01) and decrease it once it looked like the validation accuracy started to plateau, indicating overfitting. We did not want to keep training when the validation accuracy plateaus despite the training loss still decreases because then we would be overfitting on the trianing set, giving us worse results on the test set. 

## Results

// TODO

## Google Colab Notebooks

[ResNet 18](https://colab.research.google.com/drive/18vP0-6dcLXze9VGNS6pTGcW3ZJIux6FQ?usp=sharing)

[ResNet 50](https://colab.research.google.com/drive/13eX8QM6MuxEXXn8FQTLQcfumRopswTni?usp=sharing)

[ResNet 152](https://colab.research.google.com/drive/1q1buEiXvxgRqeBKwVrtbR2FCnOFZPwp0?usp=sharing)

## References

[CSE 455 Tutorial 3](https://colab.research.google.com/drive/1EBz4feoaUvz-o_yeMI27LEQBkvrXNc_4#scrollTo=X7IHgrsqd-W0)

[Cse 455 Tutorial 4](https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav#scrollTo=yRzPDiVzsyGz)

[ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)

[Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
