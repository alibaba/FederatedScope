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
CMAKE_BINARY_DIR = /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32

# Include any dependencies generated for this target.
include CMakeFiles/testTrain.out.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/testTrain.out.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testTrain.out.dir/flags.make

CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o: CMakeFiles/testTrain.out.dir/flags.make
CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o: ../../../tools/cpp/testTrain.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o -c /home/admin/gaodawei.gdw/FSDevice/MNN/tools/cpp/testTrain.cpp

CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.i"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/admin/gaodawei.gdw/FSDevice/MNN/tools/cpp/testTrain.cpp > CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.i

CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.s"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/admin/gaodawei.gdw/FSDevice/MNN/tools/cpp/testTrain.cpp -o CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.s

CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o.requires:

.PHONY : CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o.requires

CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o.provides: CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o.requires
	$(MAKE) -f CMakeFiles/testTrain.out.dir/build.make CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o.provides.build
.PHONY : CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o.provides

CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o.provides.build: CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o


# Object files for target testTrain.out
testTrain_out_OBJECTS = \
"CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o"

# External object files for target testTrain.out
testTrain_out_EXTERNAL_OBJECTS =

testTrain.out: CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o
testTrain.out: CMakeFiles/testTrain.out.dir/build.make
testTrain.out: tools/train/libMNNTrain.so
testTrain.out: libMNN_Express.so
testTrain.out: libMNN.so
testTrain.out: CMakeFiles/testTrain.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testTrain.out"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testTrain.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testTrain.out.dir/build: testTrain.out

.PHONY : CMakeFiles/testTrain.out.dir/build

CMakeFiles/testTrain.out.dir/requires: CMakeFiles/testTrain.out.dir/tools/cpp/testTrain.cpp.o.requires

.PHONY : CMakeFiles/testTrain.out.dir/requires

CMakeFiles/testTrain.out.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testTrain.out.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testTrain.out.dir/clean

CMakeFiles/testTrain.out.dir/depend:
	cd /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/admin/gaodawei.gdw/FSDevice/MNN /home/admin/gaodawei.gdw/FSDevice/MNN /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32 /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32 /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/CMakeFiles/testTrain.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testTrain.out.dir/depend

