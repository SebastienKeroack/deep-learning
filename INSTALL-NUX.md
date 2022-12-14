<a id='Releases' />

## Releases
Build and tested using:

<a id='Releases-Linux' />

#### &nbsp;&nbsp;&nbsp;&nbsp;Ubuntu 20.04
Release | Branch / Tag                                               | Adept | Boost  | Eigen | {fmt} | Clang  | GNU  
------- | ---------------------------------------------------------- | ----- | ------ | ----- | ----- | ------ | -----
Nightly | [main](https://github.com/SebastienKeroack/deep-learning)  | 2.1.1 | 1.80.0 | 3.4.0 | 9.1.1 | 15.0.6 | 9.0.4

<a id='Installation' />

## Installation
### Requirements

* [Adept: A combined automatic differentiation and array library for C++](http://www.met.reading.ac.uk/clouds/adept/)
* [Boost: "...one of the most highly regarded and expertly designed C++ library projects in the world."](https://www.boost.org/)
* [Eigen: C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.](https://eigen.tuxfamily.org/)
* [{fmt}: Open-source formatting library providing a fast and safe alternative to C stdio and C++ iostreams.](https://github.com/fmtlib/fmt)

<a id='Installation-Linux' />

#### &nbsp;&nbsp;&nbsp;&nbsp;Ubuntu 20.04
Follow the directives below to be able to build the main branch:
```shell
$ sudo apt update
$ sudo apt install build-essential cmake libsystemd-dev
```

Install __Boost__:
```shell
$ cd ~/Downloads
$ VER=1.80.0 && VER_=1_80_0
$ wget https://boostorg.jfrog.io/artifactory/main/release/${VER}/source/boost_${VER_}.tar.gz
$ tar -xvzf boost_${VER_}.tar.gz
$ sudo mv boost_${VER_}/boost /usr/local/include/boost
$ rm -rf boost_${VER_}
$ rm boost_${VER_}.tar.gz
```

Install __{fmt}__:
```shell
$ cd ~/Downloads
$ git clone https://github.com/fmtlib/fmt.git
$ cd fmt && mkdir .build && cd .build
$ cmake ..
$ sudo make install
$ cd ../.. && rm -rf fmt
```

Install __Eigen__:
```shell
$ cd ~/Downloads
$ VER=3.4.0
$ wget https://gitlab.com/libeigen/eigen/-/archive/${VER}/eigen-${VER}.tar.gz
$ tar -xvzf eigen-${VER}.tar.gz
$ cd eigen-${VER} && mkdir .build && cd .build
$ cmake ..
$ sudo cmake --build . --target install
$ cd ../.. && rm -rf eigen-${VER}
$ rm eigen-${VER}.tar.gz
```

Install __Adept__ (optional):
```shell
$ cd ~/Downloads
$ git clone https://github.com/rjhogan/Adept-2.git
$ cd Adept-2
$ sudo apt install autoconf libtool
$ autoreconf -i
$ ./configure CXXFLAGS="-g -O3 -march=native"
$ make && sudo make install
$ sudo ldconfig
$ cd .. && rm -rf Adept-2
```

Install __GoogleTest__ (optional):
```shell
$ cd ~/Downloads
$ git clone https://github.com/google/googletest.git -b release-1.12.1
$ cd googletest && mkdir .build && cd .build
$ cmake ..
$ make ..
$ sudo make install
$ cd ../.. && rm -rf googletest
```

Build __Deep Learning__:
```shell
$ cd ~/Downloads
$ git clone https://github.com/SebastienKeroack/deep-learning.git
$ cd deep-learning && mkdir .build && cd .build
# -DCMAKE_BUILD_TYPE={Debug,Release}
```
- A) Build w/ GoogleTest:
  ```shell
  $ cmake -DCMAKE_BUILD_TYPE=Release ..
  $ cmake --build .
  $ ctest --test-dir test
  ```
- B) Build w/o GoogleTest:
  ```shell
  $ cmake -DNO_TESTS=ON -DCMAKE_BUILD_TYPE=Release ..
  $ cmake --build .
  ```

Optionally you can also __install__ the library to be able to import it to your own project:
```shell
$ sudo cmake --target install
```

Finally, you can __run__ the program using the commands below:
```shell
$ cd x64/Release
$ ./deep-learning_0-2_x64.out --save --load
```