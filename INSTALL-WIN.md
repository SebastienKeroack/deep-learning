<a id='Releases' />

## Releases
Build and tested using:

<a id='Releases-Windows' />

#### &nbsp;&nbsp;&nbsp;&nbsp;Windows 11
Release | Branch / Tag                                              | Adept | Boost  | Eigen | {fmt} | MSVC 
------- | --------------------------------------------------------- | ----- | ------ | ----- | ----- | -----
Nightly | [main](https://github.com/SebastienKeroack/deep-learning) | 2.1.1 | 1.80.0 | 3.4.0 | 9.1.1 | 19.34

<a id='Installation' />

## Installation
### Requirements

* [Adept: A combined automatic differentiation and array library for C++](http://www.met.reading.ac.uk/clouds/adept/)
* [Boost: "...one of the most highly regarded and expertly designed C++ library projects in the world."](https://www.boost.org/)
* [Eigen: C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.](https://eigen.tuxfamily.org/)
* [{fmt}: Open-source formatting library providing a fast and safe alternative to C stdio and C++ iostreams.](https://github.com/fmtlib/fmt)

<a id='Installation-Windows' />

#### &nbsp;&nbsp;&nbsp;&nbsp;Windows
Follow the directives below to be able to build the main branch:
 1. Download and install [VS 2022](https://visualstudio.microsoft.com/fr/downloads/).
 2. Add `Desktop development with C++`.
 3. Add `Linux and embedded development with C++` (optional).
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
  $ del "$BOOST_NAME.zip"
  ```

Install __{fmt}__:
- Open `Developer Command Prompt for VS 2022` as Administrator:
  ```shell
  $ cd %USERPROFILE%\\Downloads
  $ git clone https://github.com/fmtlib/fmt.git
  $ cd fmt && mkdir .build && cd .build && cmake ..
  $ cmake --build . --target install --config Release
  $ cd ../.. && rmdir /s /q fmt
  ```

Install __Eigen__:
- Download [Eigen](https://eigen.tuxfamily.org/) from the main page.
- Open `Developer Command Prompt for VS 2022` as Administrator:
  ```shell
  $ cd %USERPROFILE%\\Downloads
  $ SET EIGEN_NAME=eigen-3.4.0
  $ tar -xvf "%EIGEN_NAME%.zip"
  $ cd %EIGEN_NAME% && mkdir .build && cd .build && cmake ..
  $ cmake --build . --target install
  $ cd ../.. && rmdir /s /q %EIGEN_NAME%
  ```

Install __Adept__ (optional):
- Download [Adept](http://www.met.reading.ac.uk/clouds/adept/download.html).
- Open `Windows PowerShell` as Administrator:
  ```shell
  $ cd ~/Downloads
  $ $ADEPT_NAME="adept-2.1.1"
  $ $ADEPT_DIRECTORY="C:\\Program Files (x86)\\Adept"
  $ New-Item $ADEPT_DIRECTORY -ItemType Directory
  $ tar -xvzf "$ADEPT_NAME.tar.gz" -C $ADEPT_DIRECTORY
  $ del "$ADEPT_NAME.tar.gz"
  ```

Install __GoogleTest__ (optional):
- Open `Developer Command Prompt for VS 2022` as Administrator:
  ```shell
  $ cd %USERPROFILE%\\Downloads
  $ git clone https://github.com/google/googletest.git -b release-1.12.1
  $ cd googletest && mkdir .build && cd .build
  $ cmake -Dgtest_force_shared_crt=ON ..
  $ cmake --build . --target install --config Debug
  $ SET WORKDIR=%cd%
  $ cd "C:\\Program Files (x86)\\googletest-distribution\\lib"
  $ rename gmock.lib gmockd.lib
  $ rename gmock_main.lib gmock_maind.lib
  $ rename gtest.lib gtestd.lib
  $ rename gtest_main.lib gtest_maind.lib
  $ cd %WORKDIR%
  $ cmake --build . --target install --config Release
  $ cd ../.. && rmdir /s /q googletest
  ```

Build __Deep Learning__:
- Open `Developer Command Prompt for VS 2022`:
- Clone the repository:
  ```shell
  $ cd %USERPROFILE%\\Downloads
  $ git clone https://github.com/SebastienKeroack/deep-learning.git
  $ cd deep-learning
  ```
- A) Build using __cmake__:
  ```shell
  $ mkdir .build && cd .build
  # --config {Debug,Release}
  ```
  - a) Build w/ GoogleTest:
    ```shell
    $ cmake ..
    $ cmake --build . --config Release
    $ ctest --test-dir test
    ```
  - b) Build w/o GoogleTest:
    ```shell
    $ cmake -DNO_TESTS=ON ..
    $ cmake --build . --config Release
    ```
  Optionally you can also __install__ the library to be able to import it to your own project:
  ```shell
  $ cmake --target install
  ```
- B) Build using __MSBuild__:
  ```shell
  # /p:Configuration={Debug,Release}
  $ msbuild deep-learning-win.sln /t:Build /p:Platform=x64;Configuration=Release
  ```

Finally, you can run the program using the commands below:
```shell
$ cd x64/Release
$ deep-learning_0-2_x64.exe --save --load
```