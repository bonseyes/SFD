### Steps to setup Caffe SSD on your system

1.  To use SSD, first clone the SSD repo into what we are going to call `$SFD_ROOT`:

    ```
    cd $SFD_ROOT
    git clone https://github.com/weiliu89/caffe.git
    cd caffe
    git checkout ssd
    ```

    We will call the directory that you cloned Caffe into `$CAFFE_ROOT`

2. Make sure you have cmake, make, gcc-6 (or lower) installed. We recommend to use CUDA 9.0. If you choose that version, we recommend that you also install CUDNN version 7.0.4, but other versions might work as well.

3. Follow the official [Caffe instructions](http://caffe.berkeleyvision.org/installation.html) to install all the required dependencies in your system in order to compile Caffe from scratch.

4. You can compile Caffe either using Make or CMake. We will use CMake in this guide as it is faster to configure. You can also use `ccmake` to define the libraries in an interactive way. Before you execute `cmake`, make the following changes:

    - Change a line in a file if you use CUDA 9.0. Inside the folder `cmake` that exists in `$CAFFE_ROOT`, modify the file: `Cuda.cmake` as follows
        remove this line: `set(Caffe_known_gpu_archs "20 21(20) 30 35 50 52 61")`
        and put instead: `set(Caffe_known_gpu_archs "30 35 50 52 61")`

5. Now generate all the files to compile Caffe by running `cmake` or `ccmake`:

    ```
    ## Option 1: cmake
    cd $CAFFE_ROOT
    mkdir build && cd build
    cmake -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -DCMAKE_CXX_COMPILER=/usr/bin/g++-6 -Dpython_version=3 ..

    ## Option 2: ccmake
    ccmake ..
    # Then press "c" to configure and wait for a while
    # All the dependencies and linking paths will appear, and you can change what you need. For us we change:
    #
    # BLAS: Open
    #
    # Press "t" to see more options, and then change:
    #
    # CUDA_ARCH_NAME:  All
    #
    # Then press "c" again, until this message appear in the instructions down: "Press [g] to generate and exit"
    # Then press "g"
    ```

6. Compile:

    ```
    make -j8
    # Test that everything is correctly installed
    make runtest -j8
    ```


### Troubleshooting

#### Ubuntu 

##### Anaconda issues
If you receive a huge list of errors like this during the linking phase:

```
undefined reference to `SSL_CTX_new@OPENSSL_1.0.0'
.
.
```
Make sure Caffe is not compiling against the Anaconda (or Miniconda) Python (unles you are using a virtualenv)


#### Installing other versions of Caffe

#### Caffe-jacinto
If you are using `cmake` and you face compilation errors due to OpenCV missing libraries at linking phase, such as:

```
[ 89%] Linking CXX executable upgrade_net_proto_text
[ 90%] Linking CXX executable upgrade_solver_proto_text
[ 92%] Linking CXX executable net_speed_benchmark
[ 92%] Built target test_net
[ 92%] Built target net_speed_benchmark
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::VideoCapture::set(int, double)'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::VideoWriter::write(cv::Mat const&)'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::VideoCapture::open(cv::String const&)'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::VideoCapture::release()'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::VideoCapture::operator>>(cv::Mat&)'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::VideoWriter::VideoWriter(cv::String const&, int, double, cv::Size_<int>, bool)'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::VideoCapture::isOpened() const'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::VideoCapture::get(int) const'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::VideoWriter::VideoWriter()'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::VideoCapture::open(int)'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::VideoWriter::isOpened() const'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::VideoWriter::~VideoWriter()'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::VideoCapture::~VideoCapture()'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::waitKey(int)'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::imshow(cv::String const&, cv::_InputArray const&)'
../lib/libcaffe-nv.so.0.16.4: undefined reference to `cv::VideoCapture::VideoCapture()'
```

Make sure you add the packages `imgproc videoio highgui` to the lines 74 and 77 of `$CAFFE_ROOT/cmake/Dependencies.cmake`:
```
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc videoio highgui)
```


##### GCC issues
Make sure you have gcc<=6 in your system. You can install it with:

    ```
    sudo apt-get install gcc-6
    ```

If you have access to a modern system, then you probably have a newer version of `gcc` and that can cause conflict during the compilation. To explicitly tell Caffe to compile against gcc-6, use `cmake` flags:

    ```
    cmake -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -DCMAKE_CXX_COMPILER=/usr/bin/g++-6 ...other options... 
    ```

   To find the correct path for those binaries in your system, you can run `which gcc-6`

##### CUDA issues
If your compiling fails with this error message:

    ```
    #define __CUDACC_VER__ "__CUDACC_VER__ is no longer supported. Use __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, and __CUDACC_VER_BUILD__ instead."
    ```

  Follow the instructions [here](https://github.com/BVLC/caffe/issues/5994)
