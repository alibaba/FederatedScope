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
include CMakeFiles/testModelWithDescrisbe.out.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/testModelWithDescrisbe.out.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testModelWithDescrisbe.out.dir/flags.make

CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o: CMakeFiles/testModelWithDescrisbe.out.dir/flags.make
CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o: ../../../tools/cpp/testModelWithDescrisbe.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o -c /home/admin/gaodawei.gdw/FSDevice/MNN/tools/cpp/testModelWithDescrisbe.cpp

CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.i"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/admin/gaodawei.gdw/FSDevice/MNN/tools/cpp/testModelWithDescrisbe.cpp > CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.i

CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.s"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/admin/gaodawei.gdw/FSDevice/MNN/tools/cpp/testModelWithDescrisbe.cpp -o CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.s

CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o.requires:

.PHONY : CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o.requires

CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o.provides: CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o.requires
	$(MAKE) -f CMakeFiles/testModelWithDescrisbe.out.dir/build.make CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o.provides.build
.PHONY : CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o.provides

CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o.provides.build: CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o


# Object files for target testModelWithDescrisbe.out
testModelWithDescrisbe_out_OBJECTS = \
"CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o"

# External object files for target testModelWithDescrisbe.out
testModelWithDescrisbe_out_EXTERNAL_OBJECTS =

testModelWithDescrisbe.out: CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o
testModelWithDescrisbe.out: CMakeFiles/testModelWithDescrisbe.out.dir/build.make
testModelWithDescrisbe.out: tools/train/libMNNTrain.so
testModelWithDescrisbe.out: libMNN_Express.so
testModelWithDescrisbe.out: libMNN.so
testModelWithDescrisbe.out: CMakeFiles/testModelWithDescrisbe.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testModelWithDescrisbe.out"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testModelWithDescrisbe.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testModelWithDescrisbe.out.dir/build: testModelWithDescrisbe.out

.PHONY : CMakeFiles/testModelWithDescrisbe.out.dir/build

CMakeFiles/testModelWithDescrisbe.out.dir/requires: CMakeFiles/testModelWithDescrisbe.out.dir/tools/cpp/testModelWithDescrisbe.cpp.o.requires

.PHONY : CMakeFiles/testModelWithDescrisbe.out.dir/requires

CMakeFiles/testModelWithDescrisbe.out.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testModelWithDescrisbe.out.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testModelWithDescrisbe.out.dir/clean

CMakeFiles/testModelWithDescrisbe.out.dir/depend:
	cd /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/admin/gaodawei.gdw/FSDevice/MNN /home/admin/gaodawei.gdw/FSDevice/MNN /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32 /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32 /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/CMakeFiles/testModelWithDescrisbe.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testModelWithDescrisbe.out.dir/depend

