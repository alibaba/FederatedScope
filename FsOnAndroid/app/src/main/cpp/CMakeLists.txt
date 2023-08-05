# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("fsmnn")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        fsmnn

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        native-lib.cpp
        include/MNN/tools/train/source/models/Lenet.cpp
        include/MNN/express/module/PipelineModule.cpp
        include/MNN/express/module/StaticModule.cpp
        include/MNN/express/Utils.cpp
        include/MNN/express/module/Module.cpp
        include/MNN/tools/train/source/nn/NN.cpp
        include/MNN/tools/train/source/nn/Distributions.cpp
        include/MNN/tools/train/source/datasets/FEMnistDataset.cpp
        include/MNN/tools/train/source/datasets/CelebADataset.cpp

        include/MNN/tools/converter/source/common/cli.cpp

        # fedbabu
        include/MNN/tools/train/source/optimizer/FedBabuSGD.cpp

        # twitter
        include/MNN/tools/train/source/datasets/TextDataset.cpp
        )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

set(JNI_DIR ${CMAKE_SOURCE_DIR}/../jniLibs)

add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION ${JNI_DIR}/${ANDROID_ABI}/libMNN.so)

add_library(MNNTrain SHARED IMPORTED)
set_target_properties(MNNTrain PROPERTIES IMPORTED_LOCATION ${JNI_DIR}/${ANDROID_ABI}/libMNNTrain.so)

add_library(MNN_Express SHARED IMPORTED)
set_target_properties(MNN_Express PROPERTIES IMPORTED_LOCATION ${JNI_DIR}/${ANDROID_ABI}/libMNN_Express.so)

add_library(MNNConvertDeps SHARED IMPORTED)
set_target_properties(MNNConvertDeps PROPERTIES IMPORTED_LOCATION ${JNI_DIR}/${ANDROID_ABI}/libMNNConvertDeps.so)

include_directories(${CMAKE_SOURCE_DIR}/include/)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/schema/current)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/include/)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/include/MNN)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/3rd_party/flatbuffers/include/)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/express)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/source)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/express/module)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/tools/train/source)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/tools/train/source/data)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/tools/train/source/datasets)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/tools/train/source/grad)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/tools/train/source/models)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/tools/train/source/nn)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/tools/train/source/optimizer)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/tools/train/source/parameters)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/tools/train/source/transformer)

include_directories(${CMAKE_SOURCE_DIR}/include/MNN/tools/converter/include)

# For quantization
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/tools)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/source/core)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/3rd_party/half/)
include_directories(${CMAKE_SOURCE_DIR}/include/MNN/3rd_party/protobuf/src)

target_link_libraries( # Specifies the target library.
        fsmnn

        MNN
        MNN_Express
        MNNTrain
        MNNConvertDeps

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})