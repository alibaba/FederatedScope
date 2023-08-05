#!/bin/bash

host=$1
user=$2
passwd=$3
cmd=$4

echo "$cmd"

expect <<EOF
  set timeout -1
  spawn ssh -t $user@$host bash\ -i\ -c\ "$cmd"
  expect {
    "yes/no" { send "yes\n";exp_continue }
    "password" { send "$passwd\n" }
  }
  expect eof
EOF