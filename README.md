# Deep Learning: implementations of various advances made in the field of deep learning.

A repository created in 2016, containing the implementations of various advances made in the field of deep learning. All implementations are based on information provided by their authors before mid-2019. Maybe subject to some errors. The repository has been inactive between mid-2019 and November 2022

Files inside the v1 directory/namespace are considered unintuitive/unfriendly and unstandardized.  
Otherwise, code outside the directory is written to be easier to understand from November 2022 onwards.

Parallelism is based on OpenMP directives.

## Table of contents

<a href='#Algorithms'>Algorithms</a>

<a href='#Examples'>Examples</a>

<a href='#Releases'>Releases</a>

<a href='#Installation'>Installation</a>

*  <a href='#Installation-Windows'>Windows</a>

<a href='#License'>License</a>

<a id='Algorithms'></a>
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

<a id='Examples'></a>
## Examples

End-to-end examples training.
e.g.:

*   MNIST:
    [`deep-learning-run/deep-learning/mnist.cpp`](https://github.com/SebastienKeroack/deep-learning/tree/main/deep-learning-run/deep-learning/mnist.cpp)

<a id='Releases'></a>
## Releases

Build and tested using:

Release | Branch / Tag                                               | MSVC  | Adept | Boost  | Eigen | {fmt}
------- | ---------------------------------------------------------- | ----- | ----- | ------ | ----- | -----
Nightly | [main](https://github.com/SebastienKeroack/deep-learning)  | 19.34 | 2.1.1 | 1.80.0 | 3.4.0 | 9.1.0

<a id='Installation'></a>
## Installation

### Requirements

*   [Adept: A combined automatic differentiation and array library for C++](http://www.met.reading.ac.uk/clouds/adept/)
*   [Boost: "...one of the most highly regarded and expertly designed C++ library projects in the world."](https://www.boost.org/)
*   [Eigen: C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.](https://eigen.tuxfamily.org/)
*   [{fmt}: Open-source formatting library providing a fast and safe alternative to C stdio and C++ iostreams.](https://github.com/fmtlib/fmt)

<a id='Installation-Windows'></a>
#### &nbsp;&nbsp;&nbsp;&nbsp;Windows

Follow the directives below to be able to build the main branch:
 1. Download and install [VS 2022](https://visualstudio.microsoft.com/fr/downloads/).
 2. Add `Desktop development with C++`.
 3. Add `Linux and embedded development with C++`.
 4. Download and install the latest [cmake](https://cmake.org/download/).
 5. Add `cmake` to the system `PATH` for the current user.
 
Install __Boost__:
- Download [Boost C++ Libraries](https://www.boost.org/users/download/).
- Open `Windows PowerShell` as Administrator:
```shell
$ cd ~/Downloads
$ $BOOST_NAME="boost_1_80_0"
$ $BOOST_DIRECTORY="C:\\Program Files (x86)\\Boost"
$ New-Item $BOOST_DIRECTORY -ItemType Directory
$ Expand-Archive -LiteralPath "$BOOST_NAME.zip" -DestinationPath $BOOST_DIRECTORY
```

Install __{fmt}__:
- Open `Developer Command Prompt for VS 2022` as Administrator:
```shell
$ cd %USERPROFILE%\Downloads
$ git clone https://github.com/fmtlib/fmt.git
$ cd fmt && mkdir .build && cd .build && cmake ..
$ cmake --build . --config Debug --target install
$ cmake --build . --config Release --target install
```

Install __Eigen__:
- Download [Eigen](https://eigen.tuxfamily.org/) from the main page.
- Open `Developer Command Prompt for VS 2022` as Administrator:
```shell
$ cd %USERPROFILE%\Downloads
$ SET EIGEN_NAME=eigen-3.4.0
$ tar -xvf "%EIGEN_NAME%.zip"
$ cd %EIGEN_NAME% && mkdir .build && cd .build && cmake ..
$ cmake --build . --target install
```

Install __Adept__:
- Download [Adept](http://www.met.reading.ac.uk/clouds/adept/download.html).
- Open `Windows PowerShell` as Administrator:
```shell
$ cd ~/Downloads
$ $ADEPT_NAME="adept-2.1.1"
$ $ADEPT_DIRECTORY="C:\\Program Files (x86)\\Adept"
$ New-Item $ADEPT_DIRECTORY -ItemType Directory
$ tar -xvzf "$ADEPT_NAME.tar.gz" -C $ADEPT_DIRECTORY
```

Install __Deep Learning__:
- Open `Developer Command Prompt for VS 2022`:
```shell
# Clone the repository:
$ cd %USERPROFILE%
$ git clone https://github.com/SebastienKeroack/deep-learning.git
$ cd deep-learning
$ msbuild deep-learning-win.sln -target:Build /p:Configuration=Release /p:Platform=x64
```

Finally, you can run the program using the commands below:
```shell
$ cd ../deep-learning/x64/Release/
$ ./deep-learning-run-win.exe --n_iters 20
$ ./deep-learning-run-win.exe --save --load
```

<a id='License'></a>
## License
[Apache License 2.0](LICENSE)