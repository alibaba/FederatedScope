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
include tools/train/CMakeFiles/rawDataTransform.out.dir/depend.make

# Include the progress variables for this target.
include tools/train/CMakeFiles/rawDataTransform.out.dir/progress.make

# Include the compile flags for this target's objects.
include tools/train/CMakeFiles/rawDataTransform.out.dir/flags.make

tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o: tools/train/CMakeFiles/rawDataTransform.out.dir/flags.make
tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o: ../../../tools/train/source/exec/rawDataTransform.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o"
	cd /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/tools/train && /home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o -c /home/admin/gaodawei.gdw/FSDevice/MNN/tools/train/source/exec/rawDataTransform.cpp

tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.i"
	cd /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/tools/train && /home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/admin/gaodawei.gdw/FSDevice/MNN/tools/train/source/exec/rawDataTransform.cpp > CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.i

tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.s"
	cd /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/tools/train && /home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/admin/gaodawei.gdw/FSDevice/MNN/tools/train/source/exec/rawDataTransform.cpp -o CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.s

tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o.requires:

.PHONY : tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o.requires

tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o.provides: tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o.requires
	$(MAKE) -f tools/train/CMakeFiles/rawDataTransform.out.dir/build.make tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o.provides.build
.PHONY : tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o.provides

tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o.provides.build: tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o


# Object files for target rawDataTransform.out
rawDataTransform_out_OBJECTS = \
"CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o"

# External object files for target rawDataTransform.out
rawDataTransform_out_EXTERNAL_OBJECTS =

rawDataTransform.out: tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o
rawDataTransform.out: tools/train/CMakeFiles/rawDataTransform.out.dir/build.make
rawDataTransform.out: tools/train/libMNNTrain.so
rawDataTransform.out: libMNN_Express.so
rawDataTransform.out: libMNN.so
rawDataTransform.out: tools/train/CMakeFiles/rawDataTransform.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../rawDataTransform.out"
	cd /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/tools/train && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rawDataTransform.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/train/CMakeFiles/rawDataTransform.out.dir/build: rawDataTransform.out

.PHONY : tools/train/CMakeFiles/rawDataTransform.out.dir/build

tools/train/CMakeFiles/rawDataTransform.out.dir/requires: tools/train/CMakeFiles/rawDataTransform.out.dir/source/exec/rawDataTransform.cpp.o.requires

.PHONY : tools/train/CMakeFiles/rawDataTransform.out.dir/requires

tools/train/CMakeFiles/rawDataTransform.out.dir/clean:
	cd /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/tools/train && $(CMAKE_COMMAND) -P CMakeFiles/rawDataTransform.out.dir/cmake_clean.cmake
.PHONY : tools/train/CMakeFiles/rawDataTransform.out.dir/clean

tools/train/CMakeFiles/rawDataTransform.out.dir/depend:
	cd /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/admin/gaodawei.gdw/FSDevice/MNN /home/admin/gaodawei.gdw/FSDevice/MNN/tools/train /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32 /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/tools/train /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/tools/train/CMakeFiles/rawDataTransform.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/train/CMakeFiles/rawDataTransform.out.dir/depend

