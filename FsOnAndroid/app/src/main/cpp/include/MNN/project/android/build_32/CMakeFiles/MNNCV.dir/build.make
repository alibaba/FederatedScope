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
include CMakeFiles/MNNCV.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/MNNCV.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MNNCV.dir/flags.make

CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.o: CMakeFiles/MNNCV.dir/flags.make
CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.o: ../../../source/cv/ImageProcess.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.o"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.o -c /home/admin/gaodawei.gdw/FSDevice/MNN/source/cv/ImageProcess.cpp

CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.i"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/admin/gaodawei.gdw/FSDevice/MNN/source/cv/ImageProcess.cpp > CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.i

CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.s"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/admin/gaodawei.gdw/FSDevice/MNN/source/cv/ImageProcess.cpp -o CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.s

CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.o.requires:

.PHONY : CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.o.requires

CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.o.provides: CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.o.requires
	$(MAKE) -f CMakeFiles/MNNCV.dir/build.make CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.o.provides.build
.PHONY : CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.o.provides

CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.o.provides.build: CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.o


CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.o: CMakeFiles/MNNCV.dir/flags.make
CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.o: ../../../source/cv/Matrix_CV.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.o"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.o -c /home/admin/gaodawei.gdw/FSDevice/MNN/source/cv/Matrix_CV.cpp

CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.i"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/admin/gaodawei.gdw/FSDevice/MNN/source/cv/Matrix_CV.cpp > CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.i

CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.s"
	/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=armv7-none-linux-androideabi19 --sysroot=/home/admin/gaodawei.gdw/FSDevice/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/admin/gaodawei.gdw/FSDevice/MNN/source/cv/Matrix_CV.cpp -o CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.s

CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.o.requires:

.PHONY : CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.o.requires

CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.o.provides: CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.o.requires
	$(MAKE) -f CMakeFiles/MNNCV.dir/build.make CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.o.provides.build
.PHONY : CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.o.provides

CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.o.provides.build: CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.o


MNNCV: CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.o
MNNCV: CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.o
MNNCV: CMakeFiles/MNNCV.dir/build.make

.PHONY : MNNCV

# Rule to build all files generated by this target.
CMakeFiles/MNNCV.dir/build: MNNCV

.PHONY : CMakeFiles/MNNCV.dir/build

CMakeFiles/MNNCV.dir/requires: CMakeFiles/MNNCV.dir/source/cv/ImageProcess.cpp.o.requires
CMakeFiles/MNNCV.dir/requires: CMakeFiles/MNNCV.dir/source/cv/Matrix_CV.cpp.o.requires

.PHONY : CMakeFiles/MNNCV.dir/requires

CMakeFiles/MNNCV.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MNNCV.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MNNCV.dir/clean

CMakeFiles/MNNCV.dir/depend:
	cd /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/admin/gaodawei.gdw/FSDevice/MNN /home/admin/gaodawei.gdw/FSDevice/MNN /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32 /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32 /home/admin/gaodawei.gdw/FSDevice/MNN/project/android/build_32/CMakeFiles/MNNCV.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MNNCV.dir/depend

