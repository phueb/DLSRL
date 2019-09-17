# DLSRL

## Background

This repository contains a partial implementation of the semantic role labeling model reported in He et al. (2017) using Python 3.5 and tensorflow 2.0. 

The paper describing the original implementation can be found here: 

[Deep Semantic Role Labeling: What works and what's next](https://www.aclweb.org/anthology/P17-1044)

The code is for research purpose only. 
The goal of this research project is to model early language acquisition. 
Specifically how do children comprehend utterances in terms of who is doing what to whom?

## History

The model was first built at the [NCSA GPU Hackathon](https://bluewaters.ncsa.illinois.edu/bw-hackathon-2018) in 2018.
It was updated to work with tensorflow 2.0 while working under the supervision of [Cynthia Fisher](https://psychology.illinois.edu/directory/profile/clfishe)
in the Department of Psychology at [UIUC](https://psychology.illinois.edu/). 


## Differences between this and original implementation

* LSTMs are initialized with orthogonal matrices (instead of orthonormal matrices used by He et al., 2017)
* Decoding constraints are not implemented (BIO tag sequence constraints, SRL sequence constraints)
* Control gates governing highway connections between LSTM layers are not implemented
* Recurrent dropout is not used because it prevents using cudnn acceleration of the LSTM computations in tensorflow 2.0


## Compatibility

Tested on Ubuntu 16.04, Python 3.5 and tensorflow-gpu 2.0.0rc1