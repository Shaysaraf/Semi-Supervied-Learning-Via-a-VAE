# Semi-Supervied-Learning-Via-a-VAE

semi-supervised learning via a variational autoencoder. 
In this repo we will be implementing a part of the paper "Semi-supervised Learning with
Deep Generative Models", by Kingsma et al.
We will implement the M1 scheme, as described in Algorithm1,
and detailed throughout the paper. It is based on a VAE for feature extraction,
and then a (transductive) SVM for classification.
we will implement the network suggested for MNIST, and apply it on the Fashion
MNIST data set.
At last, we will present the results for 100, 600, 1000 and 3000 labels, as
they are presented in Table 1 in the paper. 
