# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/admin/gaodawei.gdw/FSDevice/MNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_64

# Include any dependencies generated for this target.
include CMakeFiles/benchmarkExprModels.out.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/benchmarkExprModels.out.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/benchmarkExprModels.out.dir/flags.make

CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o: CMakeFiles/benchmarkExprModels.out.dir/flags.make
CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o: ../../../benchmark/benchmarkExprModels.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o -c /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/benchmarkExprModels.cpp

CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.i"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/benchmarkExprModels.cpp > CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.i

CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.s"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/benchmarkExprModels.cpp -o CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.s

CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o.requires:

.PHONY : CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o.requires

CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o.provides: CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o.requires
	$(MAKE) -f CMakeFiles/benchmarkExprModels.out.dir/build.make CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o.provides.build
.PHONY : CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o.provides

CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o.provides.build: CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o


CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o: CMakeFiles/benchmarkExprModels.out.dir/flags.make
CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o: ../../../benchmark/exprModels/GoogLeNetExpr.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o -c /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/GoogLeNetExpr.cpp

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.i"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/GoogLeNetExpr.cpp > CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.i

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.s"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/GoogLeNetExpr.cpp -o CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.s

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o.requires:

.PHONY : CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o.requires

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o.provides: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o.requires
	$(MAKE) -f CMakeFiles/benchmarkExprModels.out.dir/build.make CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o.provides.build
.PHONY : CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o.provides

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o.provides.build: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o


CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o: CMakeFiles/benchmarkExprModels.out.dir/flags.make
CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o: ../../../benchmark/exprModels/MobileNetExpr.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o -c /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/MobileNetExpr.cpp

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.i"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/MobileNetExpr.cpp > CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.i

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.s"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/MobileNetExpr.cpp -o CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.s

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o.requires:

.PHONY : CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o.requires

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o.provides: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o.requires
	$(MAKE) -f CMakeFiles/benchmarkExprModels.out.dir/build.make CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o.provides.build
.PHONY : CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o.provides

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o.provides.build: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o


CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o: CMakeFiles/benchmarkExprModels.out.dir/flags.make
CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o: ../../../benchmark/exprModels/ResNetExpr.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o -c /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/ResNetExpr.cpp

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.i"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/ResNetExpr.cpp > CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.i

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.s"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/ResNetExpr.cpp -o CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.s

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o.requires:

.PHONY : CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o.requires

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o.provides: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o.requires
	$(MAKE) -f CMakeFiles/benchmarkExprModels.out.dir/build.make CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o.provides.build
.PHONY : CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o.provides

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o.provides.build: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o


CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o: CMakeFiles/benchmarkExprModels.out.dir/flags.make
CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o: ../../../benchmark/exprModels/ShuffleNetExpr.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o -c /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/ShuffleNetExpr.cpp

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.i"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/ShuffleNetExpr.cpp > CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.i

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.s"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/ShuffleNetExpr.cpp -o CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.s

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o.requires:

.PHONY : CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o.requires

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o.provides: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o.requires
	$(MAKE) -f CMakeFiles/benchmarkExprModels.out.dir/build.make CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o.provides.build
.PHONY : CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o.provides

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o.provides.build: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o


CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o: CMakeFiles/benchmarkExprModels.out.dir/flags.make
CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o: ../../../benchmark/exprModels/SqueezeNetExpr.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o -c /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/SqueezeNetExpr.cpp

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.i"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/SqueezeNetExpr.cpp > CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.i

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.s"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/admin/gaodawei.gdw/FSDevice/MNN/benchmark/exprModels/SqueezeNetExpr.cpp -o CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.s

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o.requires:

.PHONY : CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o.requires

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o.provides: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o.requires
	$(MAKE) -f CMakeFiles/benchmarkExprModels.out.dir/build.make CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o.provides.build
.PHONY : CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o.provides

CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o.provides.build: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o


# Object files for target benchmarkExprModels.out
benchmarkExprModels_out_OBJECTS = \
"CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o" \
"CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o" \
"CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o" \
"CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o" \
"CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o" \
"CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o"

# External object files for target benchmarkExprModels.out
benchmarkExprModels_out_EXTERNAL_OBJECTS =

benchmarkExprModels.out: CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o
benchmarkExprModels.out: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o
benchmarkExprModels.out: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o
benchmarkExprModels.out: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o
benchmarkExprModels.out: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o
benchmarkExprModels.out: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o
benchmarkExprModels.out: CMakeFiles/benchmarkExprModels.out.dir/build.make
benchmarkExprModels.out: tools/train/libMNNTrain.so
benchmarkExprModels.out: libMNN_Express.so
benchmarkExprModels.out: libMNN.so
benchmarkExprModels.out: CMakeFiles/benchmarkExprModels.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable benchmarkExprModels.out"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchmarkExprModels.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/benchmarkExprModels.out.dir/build: benchmarkExprModels.out

.PHONY : CMakeFiles/benchmarkExprModels.out.dir/build

CMakeFiles/benchmarkExprModels.out.dir/requires: CMakeFiles/benchmarkExprModels.out.dir/benchmark/benchmarkExprModels.cpp.o.requires
CMakeFiles/benchmarkExprModels.out.dir/requires: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/GoogLeNetExpr.cpp.o.requires
CMakeFiles/benchmarkExprModels.out.dir/requires: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/MobileNetExpr.cpp.o.requires
CMakeFiles/benchmarkExprModels.out.dir/requires: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ResNetExpr.cpp.o.requires
CMakeFiles/benchmarkExprModels.out.dir/requires: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/ShuffleNetExpr.cpp.o.requires
CMakeFiles/benchmarkExprModels.out.dir/requires: CMakeFiles/benchmarkExprModels.out.dir/benchmark/exprModels/SqueezeNetExpr.cpp.o.requires

.PHONY : CMakeFiles/benchmarkExprModels.out.dir/requires

CMakeFiles/benchmarkExprModels.out.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/benchmarkExprModels.out.dir/cmake_clean.cmake
.PHONY : CMakeFiles/benchmarkExprModels.out.dir/clean

CMakeFiles/benchmarkExprModels.out.dir/depend:
	cd /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/admin/gaodawei.gdw/FSDevice/MNN /home/admin/gaodawei.gdw/FSDevice/MNN /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_64 /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_64 /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_64/CMakeFiles/benchmarkExprModels.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/benchmarkExprModels.out.dir/depend

