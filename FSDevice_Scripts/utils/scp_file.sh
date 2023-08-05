#!/usr/bin/expect -f

set host [lindex $argv 0]
set user [lindex $argv 1]
set passwd [lindex $argv 2]
set file_path [lindex $argv 3]
set target_path [lindex $argv 4]

set timeout -1

spawn bash -c "scp -r ${file_path} ${user}@${host}:${target_path}"

expect "*password:"

send "${passwd}\r"

expect eof
