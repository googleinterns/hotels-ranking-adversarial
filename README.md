# Hotels Ranking Adversarial

This repository demonstrates the generation of adversarial examples for textual features using the Fast Gradient Signed Method (FGSM), as described by [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) by Goodfellow et al. 

## What is an Adversarial Example?
An adversarial example is a specialized input meant to confuse a neural network. The most common example of adversarial examples is in the context of image classification. Small perturbations indistinguishable to the human eye are added to each pixel, causing the neural net to incorrectly classify the image. The example below, seen in the aforementioned paper, shows an image of a panda. Yet, after small perturbations are added the neural network incorrectly classifies the image as a gibbon.

![alt text](https://github.com/googleinterns/hotels-ranking-adversarial/blob/code-review/images/fgsm_panda.png "Panda FGSM Example")

However, adversarial examples can exist outside of the realm of image classification. In this repository we focus on the generation of adversarial examples within the context of a learning to rank (LTR) neural network. 

## Fast Gradient Signed Method
The fast gradient sign method works by using the gradients of the neural network to develop an adversarial example. For an input the method calculates the gradients of loss with respect to the input features to create a new perturbed input that maximizes the loss. This allows the neural network to be fooled even with a very small amount of perturbation. This new input is our adversarial example. This method can be encapsulated with the following equation:

![alt text](https://github.com/googleinterns/hotels-ranking-adversarial/blob/code-review/images/fgsm.png  "FGSM Equation")

where 
* x: the feature we wish to perturb
* ε: the amount of perturbation
* ∇J(θ, x, y) is the gradient of loss with respect to the feature

## Tf-Ranking Model & ANTIQUE Dataset
This repository builds upon the [TF-Ranking for sparse features tutorial](https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/examples/handling_sparse_features.ipynb), which provides a learning to rank neural network for [ANTIQUE](https://ciir.cs.umass.edu/downloads/Antique/), a question-answering dataset. Given a query, and a list of answers the neural network seeks to maximize a rank related metric (NDCG is used in this example).

ANTIQUE is a public dataset for non-factoid question answering collected over Yahoo! answers. Each question has a list of associated answers whose relevance has been ranked on a 1-5 scale. The list size may vary depending on the query so a “fixed list size” of 50 is always used. The list is truncated or padded with dummy values accordingly. The dataset is split into 2206 queries for training and 200 queries for testing. More details can be found with the technical paper found [here](https://arxiv.org/pdf/1905.08957.pdf).

The general architecture of the model (without adversarial examples) is pictured below. More details can be found within the Tf-Ranking Tutorial mentioned previously.

![alt text](https://github.com/googleinterns/hotels-ranking-adversarial/blob/code-review/images/model_achitecture.JPG  "Model Architecture Diagram")

## Setup
To run the files the following is required:
1) Python with pip (compatible with versions 3.5 - 3.8) 
  * Can be downloaded here: https://www.python.org/
2) Tensorflow
  * Can be downloaded here: https://www.tensorflow.org/install/pip
3) Tensorflow Ranking
  * Can be downloaded here: https://github.com/tensorflow/ranking
4) Matplotlib
  * Use pip to install library
```
pip install matplotlib
```

5) Dataset Files
  * Use wget to download dataset. 
```
wget -O "vocab.txt" "http://ciir.cs.umass.edu/downloads/Antique/tf-ranking/vocab.txt"
wget -O "train.tfrecords" "http://ciir.cs.umass.edu/downloads/Antique/tf-ranking/ELWC/train.tfrecords"
wget -O "test.tfrecords" "http://ciir.cs.umass.edu/downloads/Antique/tf-ranking//ELWC/test.tfrecords"
```
## Running the Code
The code can be executed by calling the following:
```
python main.py directory_name/
```
If no argument is filled in for the directory, visualizations will be saved in the current directory by default. 

Upon running the code, a query of will be displayed alongside a list of questions ranked in order of relevance. The user can then select a question to be perturbed, a reference answer, and the amount of perturbation. The reference answer serves to indicate the direction of the perturbation. For instance, if the reference answer has a higher rank than the specified answer to be perturbed, the perturbation will increase the rank of the answer. Then the model will rank the answers again, this time with the perturbed input. For comparison purposes, the model will rank the answers one last time with randomly generated (as opposed to generated with FGSM) noise to demonstrate the larger effect FGSM generated noise tends to have on the model. Then a bar plot demonstrating the rankings of both the unperturbed, FGSM-perturbed, and randomly perturbed inputs will be displayed and saved to the specified folder. Other visualizations (described below) can also be added.


## Visualizations




