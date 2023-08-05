#!/bin/bash --login

workdir=$1

echo "=========================================================="
echo "Step1: Check working directory in ${workdir} ..."

# Open work directory
if [ ! -d "${workdir}" ];then
  echo "Doesn't exist, mkdir in ${workdir}"
  mkdir -p "${workdir}"
fi

cd "${workdir}"

echo "=========================================================="
echo "Step2: Install system libraries if needed ..."

# install system libraries
apt-get update
apt install -y unzip wget openjdk-8-jdk cpu-checker qemu-kvm libvirt-bin bridge-utils virtinst virt-manager expect

echo "=========================================================="
echo "Step3: Install python libraries if needed ..."

# python libraries
pip3 install numpy argparse tqdm PyYAML


echo "=========================================================="
echo "Step4: Download android sdk and set environment paths if needed ..."

# verify emulator adb
if [ ! -d "$ANDROID_SDK_ROOT" ]; then
  echo "Install android sdk ..."
  # download android cmdline-tools
  mkdir -p "$HOME/Android/sdk/cmdline-tools/"
  echo "Download android commandline tools ..."
  wget https://dl.google.com/android/repository/commandlinetools-linux-9123335_latest.zip
  unzip -q commandlinetools-linux-9123335_latest.zip
  mv cmdline-tools "$HOME/Android/sdk/cmdline-tools/latest"
  rm -rf cmdline-tools commandlinetools-linux-9123335_latest.zip

  # import path for android
  mkdir -p $HOME/.android/avd
  echo -e "\n# for android sdk
export ANDROID_SDK_ROOT=$HOME/Android/sdk
export PATH=\$ANDROID_SDK_ROOT/emulator:\$PATH
export PATH=\$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:\$PATH
export PATH=\$ANDROID_SDK_ROOT/platform-tools:\$PATH
export ANDROID_AVD_HOME=\$HOME/.android/avd" >> ~/.bashrc

  # import into current bash
  export ANDROID_SDK_ROOT=$HOME/Android/sdk
  export PATH=$ANDROID_SDK_ROOT/emulator:$PATH
  export PATH=$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$PATH
  export PATH=$ANDROID_SDK_ROOT/platform-tools:$PATH
  export ANDROID_AVD_HOME=$HOME/.android/avd
else
  echo "Android sdk root already exists, which is $ANDROID_SDK_ROOT"
fi

echo "=========================================================="
echo "Step5: Download android sdk and set environment paths if needed ..."

# install emulator and adb
if sdkmanager --list_installed | grep emulator >/dev/null 2>&1 ; then
  echo "Emulator is already installed"
else
  echo "Install emulator ..."
  /usr/bin/expect << EOF
  set timeout -1
  spawn bash -i -c "sdkmanager \"emulator\""
  expect {
  "(y/N): " { send "y\r"; exp_continue }
  eof {}
  }
EOF
  echo "Done"
fi

if sdkmanager --list_installed | grep "platform-tools" >/dev/null 2>&1 ; then
  echo "Platform-tools is already installed"
else
  echo "Install platform-tools ..."
  /usr/bin/expect << EOF
  set timeout -1
  spawn bash -i -c "sdkmanager \"platform-tools\""
  expect {
  "(y/N): " { send "y\r"; exp_continue }
  eof {}
  }
EOF
  echo "Done"
fi

if sdkmanager --list_installed | grep "platforms;android-30" >/dev/null 2>&1 ; then
  echo "Platforms;android-30 is already installed"
else
  echo "Install platforms;android-30 ..."
  /usr/bin/expect << EOF
  set timeout -1
  spawn bash -i -c "sdkmanager \"platforms;android-30\""
  expect {
  "(y/N): " { send "y\r"; exp_continue }
  eof {}
  }
EOF
  echo "Done"
fi

if sdkmanager --list_installed | grep "system-images;android-30;google_apis;x86_64" >/dev/null 2>&1 ; then
  echo "System-images;android-30;google_apis;x86_64 is already installed"
else
  echo "Install system-images;android-30;google_apis;x86_64 ..."
  /usr/bin/expect << EOF
  set timeout -1
  spawn bash -i -c "sdkmanager \"system-images;android-30;google_apis;x86_64\""
  expect {
  "(y/N): " { send "y\r"; exp_continue }
  eof {}
  }
EOF
  echo "Done"
fi


#/usr/bin/expect << EOF
#set timeout -1
#spawn bash -i -c "sdkmanager \"emulator\" \"platform-tools\" \"platforms;android-30\" \"system-images;android-30;google_apis;x86_64\""
#expect {
#"(y/N):" { send "y\n" }
#eof {}
#}
#EOF

echo "Finish!"
echo "=========================================================="
