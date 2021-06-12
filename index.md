## Bird Kaggle Competition for CSE 455

This website describes our process for testing different resnet architectures on the birds kaggle competition from [here](https://www.kaggle.com/c/birds21sp) for the CSE 455 course at University of Washington for Spring 2021.

Group: Akshit Arora, Braxton Kinney, Frederick Huyan, Leo Liao, Nikolai Scheel

For the video project explanation, [click here](https://drive.google.com/file/d/1jbVxttSRtALSgXeP8SEECBCyfPE1Yscu/view?usp=sharing)

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

We first started by porting over the tutorial 4 code which included getting the data, training, and predicted. We slightly modified the code to get the data to split off 1000 examples for a validation set by getting 1000 random indices from the train data. For all trainings, we used a batch size of 32.

### ResNet 18

We started with ResNet 18. We first trained it for 5 epochs at a learning rate of 0.01, then evaluated the model on the validation set and plotted the results.

![image](https://user-images.githubusercontent.com/30280125/121585130-14e34880-c9e7-11eb-90e8-f2863fa5c5d1.png)

As we can see, the validation accuracy has a clear upward trend although slightly slowing down. We decided to train for 5 more epochs with the same learning rate to see if it would plataeu and overfit.

![image](https://user-images.githubusercontent.com/30280125/121585545-90dd9080-c9e7-11eb-9847-b11e890ad40a.png)

It definitely did start to plataeu and decrease which is a clear sign of overfitting because the training loss continued to decrease with each epoch. Therefore, we decided to reduce the learning rate by a factor of 10 down to 0.001 for another 5 epochs to see if we could get anymore improvement. 

![image](https://user-images.githubusercontent.com/30280125/121585736-c2565c00-c9e7-11eb-8061-b30c58e484b5.png)

At this point, it looked like the trend was still barely upwards so we decided to do another 5 epochs at 0.001 learning rate

![image](https://user-images.githubusercontent.com/30280125/121585830-e154ee00-c9e7-11eb-946f-c8189004272a.png)

We can now clearly see a downtrend we ended up taking the checkpoint 13 where the validation accuracy peaked and tested it on Kaggle to see our test accuracy. This got a score of 0.63400 which was our baseline that we wanted to beat using ResNet 50 and ResNet 152. Interestingly, our test accuracy was fairly close to our validation accuracy. 

### ResNet 50

For ResNet 50, we had the same methodology. We started off with 5 epochs with a learning rate of 0.01 to see where we could get in terms of validation accuracy. 

![image](https://user-images.githubusercontent.com/30280125/121586721-d0f14300-c9e8-11eb-9cc8-6568601653ca.png)

The results were surprisingly very simialr to ResNet 18 in the first 5 epochs, which was rather dissapointing. It also looked like it was nearly plataeuing already, so for the next 5 epochs, we trained two epochs with a learning rate of 0.01 and the next three with a learning rate of 0.001, which gave us very good results.

![image](https://user-images.githubusercontent.com/30280125/121586997-2cbbcc00-c9e9-11eb-90b0-1a13b265f612.png)

Seeing this, we decided to continue to train at a learning rate of 0.001 for another 5 epochs.

![image](https://user-images.githubusercontent.com/30280125/121587113-49580400-c9e9-11eb-894a-2be51ee2c5e4.png)

Similar to ResNet 18, there seems to be a tiny small upward trend, but otherwise plateauing so we went for a final 5 epochs at 0.001 to see if the validation accuracy would improve any further. 

![image](https://user-images.githubusercontent.com/30280125/121587225-67256900-c9e9-11eb-9af7-3a0733f79014.png)

At this point, it definitely seems like it was plataueing, so we chose checkpoint 19 as our final choise and submitted to Kaggle, giving us a test accuracy of 0.803, also very close to our validation accuracy, and much better than ResNet 18.

### ResNet 152

Finally, we tested ResNet 152, with the assumption that it would do the best. We once again started off with 5 epochs at a learning rate of 0.01.

![image](https://user-images.githubusercontent.com/30280125/121587412-a05dd900-c9e9-11eb-911c-28fde1dc1305.png)

It looks like this was already better at 5 epochs, reaching nearly 0.6 validation accuracy. Since it looks similar to ResNet 50 in that it was slightly plataeuing, we decided to do something similar where we trained for another 2 epochs at a learning rate of 0.01 then lowered it to 0.001 for the next 3.

![image](https://user-images.githubusercontent.com/30280125/121591413-51667280-c9ee-11eb-8e32-e4ce84922d08.png)

Once again the accuracy seemed to spike at epoch 7 when the learning rate is changed to 0.001, so we decided to continue training at 0.001 for another 5 epochs.

![image](https://user-images.githubusercontent.com/30280125/121592082-1b75be00-c9ef-11eb-9fcd-435a622a966e.png)

Following a similar pattern, it looked like there was still a slight upward trend so we did 5 more epochs at 0.001.

![image](https://user-images.githubusercontent.com/30280125/121592231-46f8a880-c9ef-11eb-89f5-759bece75a3e.png)

Which also clearly shows signs of plateauing. We took checkpoint 17 as our final model and submitted it to Kaggle, to get our final score of 0.832. So as expected, ResNet 152 did the best.

## Final Thoughts

As none of us were very familiar with machine learning, combined with the very long training times and trouble with Google Colab, we wanted to do more training with diferent parameters such as actually modifying some of the layers, using a different optimizer, modifying the momentum, and adding layers but simply did not have enough time, although those are may not have had a huge impact on the final result. We also wanted to retrain the model once we chose the epoch and learning rates we needed with the validation data added back in but that was taking too long and causing issues with Google Colab, although we believe that would have definitely bumped up the test acuracy a little bit due to have 1000 more training samples. 

The clear picture we took away from this was the ability for larger models to improve performance to a certain extent. At some point, smaller networks reach a point where the input has too many features for the network to model, such as in bird pictures where there are small differences across similar species, and the bias can't be minimized anymore. Thus, we need a larger and more powerful network to model all those features. However, the difference between ResNet 50 and ResNet 152 were not huge. we think we definitely could have done better across all three models if we had a lot more time to experiment. For exampe, it was strange that in ResNet 50 and ResNet 152, the validation accuracy shot up to around 80 when we decreased the learning rate to 0.001, which may indicate that we found a good minima. We may have passed a good minimum for ResNet 18 or even been stuck in local minima. Another thing we could have done is compare different networks such as VGG or InceptionNet V3, however, we wanted to focus on using the same architecture but with different numbers of layers to narrow down the effect of making a neural network deeper. 


## Google Colab Notebooks

[ResNet 18](https://colab.research.google.com/drive/18vP0-6dcLXze9VGNS6pTGcW3ZJIux6FQ?usp=sharing)

[ResNet 50](https://colab.research.google.com/drive/13eX8QM6MuxEXXn8FQTLQcfumRopswTni?usp=sharing)

[ResNet 152](https://colab.research.google.com/drive/1q1buEiXvxgRqeBKwVrtbR2FCnOFZPwp0?usp=sharing)

## References

[CSE 455 Tutorial 3](https://colab.research.google.com/drive/1EBz4feoaUvz-o_yeMI27LEQBkvrXNc_4#scrollTo=X7IHgrsqd-W0)

[Cse 455 Tutorial 4](https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav#scrollTo=yRzPDiVzsyGz)

[ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)

[Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
