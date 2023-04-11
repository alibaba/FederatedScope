# Copyright 2017 The Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/usr/bin/env bash
# YAPF + Clang formatter (if installed). This script formats all changed files from the last mergebase.
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -eo pipefail

ver=$(yapf --version)
if ! echo $ver | grep -q 0.31.0; then
    echo "Wrong YAPF version installed: 0.31.0 is required, not $ver"
    exit 1
fi

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"

ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

FLAKE8_VERSION=$(flake8 --version | awk '{print $1}')
YAPF_VERSION=$(yapf --version | awk '{print $2}')

# params: tool name, tool version, required version
tool_version_check() {
    if [[ $2 != $3 ]]; then
        echo "WARNING: FedHPOB uses $1 $3, You currently are using $2. This might generate different results."
    fi
}

tool_version_check "flake8" $FLAKE8_VERSION "4.0.1"
tool_version_check "yapf" $YAPF_VERSION "0.31.0"

# Only fetch master since that's the branch we're diffing against.
#git fetch upstream master || true
git fetch origin master || true

YAPF_FLAGS=(
    '--style' "$ROOT/.style.yapf"
    '--recursive'
    '--parallel'
)

YAPF_EXCLUDES=(
    '--exclude' 'scripts/*'
)

# Format specified files
format() {
    yapf --in-place "${YAPF_FLAGS[@]}" -- "$@"
}

# Format files that differ from main branch. Ignores dirs that are not slated
# for autoformat yet.
format_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause yapf to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only format files that
    # exist on both branches.
    #MERGEBASE="$(git merge-base upstream/master HEAD)"
    MERGEBASE="$(git merge-base origin/master HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' | xargs -P 5 \
             yapf --in-place "${YAPF_EXCLUDES[@]}" "${YAPF_FLAGS[@]}"
        if which flake8 >/dev/null; then
            git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' | xargs -P 5 \
                 flake8 --inline-quotes '"' --no-avoid-escape --ignore=C408,E121,E123,E126,E226,E24,E704,W503,W504,W605
        fi
    fi

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.pyx' '*.pxd' '*.pxi' &>/dev/null; then
        if which flake8 >/dev/null; then
            git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.pyx' '*.pxd' '*.pxi' | xargs -P 5 \
                 flake8 --inline-quotes '"' --no-avoid-escape --ignore=C408,E121,E123,E126,E226,E24,E704,W503,W504,W605
        fi
    fi
}

# Format all files, and print the diff to stdout for travis.
format_all() {
    yapf --diff "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" fedhpob
    #yapf --in-place "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" fedhpob
}

# This flag formats individual files. --files *must* be the first command line
# arg to use this option.
if [[ "$1" == '--files' ]]; then
    format "${@:2}"
    # If `--all` is passed, then any further arguments are ignored and the
    # entire python directory is formatted.
elif [[ "$1" == '--all' ]]; then
    format_all
else
    # Format only the files that changed in last commit.
    format_changed
fi

if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted changed files. Please review and stage the changes.'
    echo 'Files updated:'
    echo

    git --no-pager diff --name-only

    exit 1
fi
