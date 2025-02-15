# Trying out different ways to do stable diffusion

 

The purpose is to create a new human face from 2 source faces

### Dataset Aquisition
In this implementation I found 2 human models with many face pictures, created a dataset of 30 pictures each


### Dataset preprocessing 

Pictures were selected to have the ratio 3:4 and those who were larger than 960 X 1280 were shrunk to this size

### Implementation notes:

I tried more approaches with yielded low quality \
In the end the best approach was to use a pre-trained model, here I used stabilityai/stable-diffusion-xl-base-1.0

I tried fine tuning all the Unet parameters using the Pytorch Lighning library \
But in the end the best best results were yielded using Lora and also training the text encoders\
From discussions on reddit I found out the model should do 100 steps on each image for best results


