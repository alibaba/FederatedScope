# Install script for directory: /home/admin/gaodawei.gdw/FSDevice/MNN

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/MNN" TYPE FILE FILES
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/MNNDefine.h"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/Interpreter.hpp"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/HalideRuntime.h"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/Tensor.hpp"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/ErrorCode.hpp"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/ImageProcess.hpp"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/Matrix.h"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/Rect.h"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/MNNForwardType.h"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/AutoTime.hpp"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/MNNSharedContext.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/MNN/expr" TYPE FILE FILES
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/expr/Expr.hpp"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/expr/ExprCreator.hpp"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/expr/MathOp.hpp"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/expr/NeuralNetWorkOp.hpp"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/expr/Optimizer.hpp"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/expr/Executor.hpp"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/expr/Module.hpp"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/expr/NeuralNetWorkOp.hpp"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/expr/ExecutorScope.hpp"
    "/home/admin/gaodawei.gdw/FSDevice/MNN/include/MNN/expr/Scope.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/libMNN.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/express/cmake_install.cmake")
  include("/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/tools/train/cmake_install.cmake")
  include("/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/tools/converter/cmake_install.cmake")
  include("/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/tools/cv/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
