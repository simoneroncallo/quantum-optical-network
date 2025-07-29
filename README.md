# Quantum optical network

This repository contains the simulation of a quantum optical shallow network implemented through the Hong-Ou-Mandel effect. The main code can be found in `quantumNetwork.ipynb`, implemented with arbitrary number of neurons. Training and validation data are prepared in `dataprocessing.py`, using the [MNIST](https://www.tensorflow.org/datasets/catalog/mnist?hl=it), the [Fashion-MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist), and the [CIFAR-10](https://www.tensorflow.org/datasets/catalog/cifar10?hl=it) datasets for binary classification tasks. A classical analogue is reported in `classicalNetwork.ipynb`, with the possibility to add constraints given by the normalization and the positivity conditions. Dependencies: `tensorflow 2.18.0` and `keras 3.8.0`.

[![arXiv](https://img.shields.io/badge/arXiv-2507.21036-b31b1b.svg)](https://arxiv.org/abs/2507.21036)
<a target="_blank" href="https://colab.research.google.com/github/simoneroncallo/quantum-optical-network/blob/main/quantumNetwork.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> <br>
  
Contributors: Angela Rosy Morgillo [@MorgilloR](https://github.com/MorgilloR) and Simone Roncallo [@simoneroncallo](https://github.com/simoneroncallo) <br>
Reference: Simone Roncallo, Angela Rosy Morgillo, Chiara Macchiavello, Lorenzo Maccone and Seth Lloyd <i>“Quantum optical shallow networks”</i> (In preparation, 2025)
