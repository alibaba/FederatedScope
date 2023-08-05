#!/bin/bash
devices=`adb devices`

for device in $devices; do
	if [[ "$device" =~ "emulator-" ]]; then
		adb -s $device emu kill
		echo $device removed
	fi
done

# TODO: take the name of basic avd as input
ps -ef|grep test_x86|grep -v grep|awk '{print $2}'|xargs kill -9

free -g

rm -rf /tmp/android-root

echo "All Done."