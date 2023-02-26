# Deep Learning: implementations of various advances made in the field of deep learning.

A library created in 2016, containing the implementations of various advances made in the field of deep learning. All implementations are based on research papers published before mid-2019. Technical information is based on an open access repository of electronic preprints and postprints. The repository has been inactive between mid-2019 and November 2022

Files inside the v1 directory/namespace are considered unintuitive/unfriendly and unstandardized.  
Otherwise, code outside the directory is written to be easier to understand from November 2022 onwards.

Parallelism based on OpenMP directives.

## Table of contents
* <a href='#Algorithms'>Algorithms</a>
* <a href='#Examples'>Examples</a>
* Releases
  * [Windows](https://github.com/SebastienKeroack/deep-learning/blob/main/INSTALL-WIN.md#Releases-Windows "Releases, Windows")
  * [Linux](https://github.com/SebastienKeroack/deep-learning/blob/main/INSTALL-NUX.md#Releases-Linux "Releases, Linux")
* Installation
  * [Windows](https://github.com/SebastienKeroack/deep-learning/blob/main/INSTALL-WIN.md#Installation-Windows "Installation on Windows")
  * [Linux](https://github.com/SebastienKeroack/deep-learning/blob/main/INSTALL-NUX.md#Installation-Linux "Installation on Linux")
* <a href='#Import'>Import</a>
* <a href='#License'>License</a>

<a id='Algorithms' />

## Algorithms
Presently the following algorithms and more are available under DL:

#### &nbsp;&nbsp;&nbsp;&nbsp;Activations:
* [Leaky ReLU: __Rectifier Nonlinearities Improve Neural Network Acoustic Models__ Andrew L. Maas et al, 2013](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)
* [ELU: __Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)__ Djork-Arné Clevert et al., 2015](https://arxiv.org/abs/1511.07289)
* [SELU: __Self-Normalizing Neural Networks__ Günter Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
* [ISRLU: __Improving Deep Learning by Inverse Square Root Linear Units (ISRLUs)__ Brad Carlile et al, 2017](https://arxiv.org/abs/1710.09967)
* etc...

#### &nbsp;&nbsp;&nbsp;&nbsp;Initializers:
* [Glorot: __Understanding the difficulty of training deep feedforward neural networks__ Xavier Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
* [Orthogonal: __Exact solutions to the nonlinear dynamics of learning in deep linear neural networks__ Andrew Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
* [LSUV: __All you need is a good init__ Dmytro Mishkin et al, 2015](https://arxiv.org/abs/1511.06422)
* etc...

#### &nbsp;&nbsp;&nbsp;&nbsp;Layers:
* [LSTM: __Long Short-Term Memory layer__ Sepp Hochreiter et al., 1997](http://www.bioinf.jku.at/publications/older/2604.pdf)
* [Batch normalization: __Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift__ Sergey Ioffe et al., 2015](https://arxiv.org/abs/1502.03167)
* [Residual: __Deep Residual Learning for Image Recognition__ Kaiming He et al., 2015](https://arxiv.org/abs/1512.03385)
* [Zoneout: __Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations__ David Krueger et al., 2016](https://arxiv.org/abs/1606.01305)
* [Batch renormalization: __Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models__ Sergey Ioffe et al., 2017](https://arxiv.org/abs/1702.03275)
* [IndRNN: __Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN__ Shuai Li et al, 2018](https://arxiv.org/abs/1803.04831)
* [ShakeDrop: __ShakeDrop Regularization for Deep Residual Learning__ Yoshihiro Yamada et al, 2018](https://arxiv.org/abs/1802.02375)
* [UOut: __Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift__ Xiang Li et al, 2018](https://arxiv.org/abs/1801.05134)
* etc...

#### &nbsp;&nbsp;&nbsp;&nbsp;Optimizers:
* [Adam: __Adam: A Method for Stochastic Optimization__ Diederik P. Kingma et al., 2014](https://arxiv.org/abs/1412.6980)
* [AdaMax: __Adam: A Method for Stochastic Optimization__ Diederik P. Kingma et al., 2014](https://arxiv.org/abs/1412.6980)
* [Rprop: __Adapting Resilient Propagation for Deep Learning__ Alan Mosca et al., 2015](https://arxiv.org/abs/1509.04612)
* [AdaBound: __Adaptive Gradient Methods with Dynamic Bound of Learning Rate__ Liangchen Luo et al., 2018](https://openreview.net/forum?id=Bkg3g2R9FX)
* [AMSBound: __Adaptive Gradient Methods with Dynamic Bound of Learning Rate__ Liangchen Luo et al., 2018](https://openreview.net/forum?id=Bkg3g2R9FX)
* [AMSGrad: __On the Convergence of Adam and Beyond__ Sashank J. Reddi et al., 2018](https://openreview.net/forum?id=ryQu7f-RZ)
* [NosAdam: __Nostalgic Adam: Weighting more of the past gradients when designing the adaptive learning rate__ Haiwen Huang et al., 2018](https://arxiv.org/abs/1805.07557)
* etc...

<a id='Examples' />

## Examples
End-to-end examples training.
e.g.:
  * MNIST:
    [`run/deep-learning/mnist.cpp`](https://github.com/SebastienKeroack/deep-learning/tree/main/run/deep-learning/mnist.cpp)

<a id='Import' />

## Import
If you have installed the library you can import it to your project by adding these lines inside your cmake file.
```
find_package (DEEP-LEARNING CONFIG REQUIRED)
target_include_directories (${PROJECT_NAME} PRIVATE ${DEEPLEARNING_INCLUDE_DIR})
target_link_libraries (${PROJECT_NAME} PRIVATE DL::DEEPLEARNING)
```

<a id='License' />

## License
[Apache License 2.0](LICENSE)
