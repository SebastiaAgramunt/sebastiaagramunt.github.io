---
title: C++ compile and link external library
author: sebastia
date: 2025-01-19 7:05:00 +0800
categories: [C++]
tags: [computer science]
pin: true
toc: true
render_with_liquid: false
math: true
---

As mentioned in other posts C++ is a rich programming language that has been out there for a while, it's predecessor C was created in the 70s and C++ in 1979. Naturally many projects flourished using this language and therefore we have lots of resources out there to use, among them [OpenCV](https://opencv.org/), [Boost](https://www.boost.org/), [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) or [cBLAS](https://www.gnu.org/software/gsl/doc/html/cblas.html). In this post we will see an example on how to compile link and use OpenCV in a custom C++ program.

The code to generate the following can be found in the repository [blogging-code](https://github.com/SebastiaAgramunt/blogging-code) in the subdirectory [cpp-compile-link-external-lib](https://github.com/SebastiaAgramunt/blogging-code/tree/main/cpp-compile-link-external-lib).


## File structure

If I have to use a library that is specific to a project I prefer to have the structure/library inside the same project then the project is self contained. I normally create a bash script that downloads and compiles the library inside a directory called `external`. The structure would be

```
.
├── external
│   └── install-opencv.sh
├── main.cpp
└── scripts
    ├── compile-run.sh
    └── download-img.sh
```

where the script `install-opencv.sh` contains the routines for download and build the library, the source `main.cpp` the example using opencv routines, `compile-run.sh` the instructions to compile the `main.cpp` and `download-img.sh` the code to download the example image.

## Installing Opencv

The script `install-opencv.sh` will install the opencv library version `4.10.0`

```bash
#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})
OPENCV_VERSION=4.10.0


# library installed in this directory/lib
LIBDIR=${THIS_DIR}/lib

# download and untar
wget https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz -O ${THIS_DIR}/opencv.tar.gz
cd ${THIS_DIR} && tar -xzf ${THIS_DIR}/opencv.tar.gz

# build the library
cd ${THIS_DIR}/opencv-${OPENCV_VERSION}
mkdir -p build && cd build

cmake -D CMAKE_BUILD_TYPE=Release \
	-D CMAKE_INSTALL_PREFIX=$LIBDIR/opencv \
	-D BUILD_EXAMPLES=ON ..

make -j$(nproc)
make install

# remove temporary files
cd ${THIS_DIR} && rm -rf opencv-${OPENCV_VERSION}
cd ${THIS_DIR} && rm -rf opencv.tar.gz

```

will install opencv in `external/lib/opencv` for us to use. The includes are in `opencv/include`, the binaries in `opencv/bin` and the compiled libraries in `opencv/lib`.

Simply run with

```bash
chmod +x external/install-opencv.sh
./external/install-opencv.sh
```

And the library will be compiled and installed in `external` directory.
## Download image

To run the example we need to download an image, I've chosen a free one from wikimedia commons. Write the script `download-img.sh` with the following:

```bash
#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})

mkdir ${ROOT_DIR}/img
wget "https://upload.wikimedia.org/wikipedia/commons/2/28/20100723_Miyajima_4904.jpg" -O ${ROOT_DIR}/img/raw_img.jpeg
```

execute it with

```bash
chmod +x scripts/download-img.sh
./scripts/download-img.sh
```

And this will download the image in `img/raw_img.jpeg`.

## The main

The example consists on a small main that loads a file, converts it to grayscale and finally blurs it with a gaussian blur.  Here is the code:

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    // Check if the user provided an argument
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Get the image path from the command-line argument
    std::string imagePath = argv[1];

    // Read the image
    cv::Mat image = cv::imread(imagePath);

    // Check if the image was successfully loaded
    if (image.empty()) {
        std::cerr << "Error: Unable to load image at " << imagePath << std::endl;
        return -1;
    }

    // Convert the image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Apply Gaussian blur
    cv::Mat blurredImage;
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(15, 15), 5.0);

    // Display the original and processed images
    cv::imshow("Original Image", image);
    cv::imshow("Blurred Image", blurredImage);

    // Wait for a key press
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);

    // // Save the processed image
    std::string outputPath = "blurred_image.jpg";
    cv::imwrite(outputPath, blurredImage);
    std::cout << "Processed image saved to " << outputPath << std::endl;

    return 0;
}

```

Copy these contents into the `main.cpp` file. 

## Compile and run the example

Now we have all the important files written but we still need to compile and run the example, write the following contents to the `scripts/compile_run.sh` file

```bash
#!/bin/bash

THIS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(dirname ${THIS_DIR})

recreate_dirs(){
    # removing build directory
    echo "Removing ${ROOT_DIR}/build and recreating..."
    rm -rf ${ROOT_DIR}/build
    mkdir ${ROOT_DIR}/build

    # creating directories for the build
    mkdir ${ROOT_DIR}/build/obj
    mkdir ${ROOT_DIR}/build/bin
    mkdir ${ROOT_DIR}/build/lib
}

compile_exec(){
    recreate_dirs

    # compile to objects
    echo "Compiling objects for executable..."
    g++ -std=c++17 -I${ROOT_DIR}/external/lib/opencv/include/opencv4 -c ${ROOT_DIR}/main.cpp -o ${ROOT_DIR}/build/obj/main.o

    # link all the objects
    g++ ${ROOT_DIR}/build/obj/main.o \
        -I${ROOT_DIR}/external/lib/opencv/include/opencv4 \
        -L${ROOT_DIR}/external/lib/opencv/lib \
        -lopencv_core \
        -lopencv_highgui \
        -lopencv_imgproc \
        -lopencv_imgcodecs \
        -Wl,-rpath,external/lib/opencv/lib \
        -o ${ROOT_DIR}/build/bin/main
}

croak(){
    echo "[ERROR] $*" > /dev/stderr
    exit 1
}

main(){
    if [[ -z "$TASK" ]]; then
        croak "No TASK specified."
    fi
    echo "[INFO] running $TASK $*"
    $TASK "$@"
}

main "$@"

```

See the compilation phase, to generate the main object we just need the includes from the library, those are in `external/lib/opencv/include/opencv4`. In the linking part we also add the includes but also the libraries with the directory `external/lib/opencv/lib`. Then with the flag `-l` we specify the libraries. Just to name one `opencv_imgcodecs` can be found in `external/lib/opencv/lib/libopencv_imgcodecs.dylib`. In my case since I work in a MacOS the libraries are `dylib` extension, this is dynamic library for MacOS. Compile the code with

```bash
chmod +x ./scripts/compile-run.sh

export TASK=compile_exec
./scripts/compile-run.sh 
```

This will create the usual `build` directory with the subdirectories and place the executable in the `bin`.

Run with the argument the filepath of the image you downloaded in the previous section

```bash
./build/bin/main img/raw_img.jpeg
```

## Conclusions

We downloaded `opencv` library and compiled it from scratch. Placed the library inside our project directory. Next we used the library into a custom program to convert an image to black and white and blur it with a gaussian filter. The steps to use another library like Boost are similar, download the source code, compile it and then use the flags `-I` to add the includes of your library and `-L` to indicate where are the compiled libraries and `-l` to tell which libraries should be considered in the linking step.