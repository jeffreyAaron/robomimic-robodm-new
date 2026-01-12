#!/usr/bin/env bash
# YAPF formatter, adapted for robodm project.
#
# Usage:
#    # Do work and commit your work.

#    # Format files that differ from origin/main.
#    bash format.sh

#    # Check formatting without making changes (for CI)
#    bash format.sh --check

#    # Format all files
#    bash format.sh --all

#    # Commit changed files with message 'Run yapf and pylint'
#
#
# YAPF + Black formatter. This script formats all changed files from the last mergebase.
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -eo pipefail

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

# Parse command line arguments
CHECK_ONLY=false
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --files)
            # Keep existing behavior for --files
            break
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--check] [--all] [--files file1 file2 ...]"
            exit 1
            ;;
    esac
done

# Check if tools are installed before getting versions
check_tool_installed() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed. Please install development dependencies."
        echo "You can install them with: pip install yapf black isort mypy pylint flake8"
        exit 1
    fi
}

check_tool_installed "yapf"
check_tool_installed "black"
check_tool_installed "isort"
check_tool_installed "mypy"
check_tool_installed "pylint"

YAPF_VERSION=$(yapf --version | awk '{print $2}')
BLACK_VERSION=$(black --version | head -n 1 | awk '{print $2}')
ISORT_VERSION=$(isort --version | head -n 1 | awk '{print $2}')
MYPY_VERSION=$(mypy --version | awk '{print $2}')
PYLINT_VERSION=$(pylint --version | head -n 1 | awk '{print $2}')

echo "Using formatting tools:"
echo "  yapf: $YAPF_VERSION"
echo "  black: $BLACK_VERSION"
echo "  isort: $ISORT_VERSION"
echo "  mypy: $MYPY_VERSION"
echo "  pylint: $PYLINT_VERSION"
echo

YAPF_FLAGS=(
    '--recursive'
    '--parallel'
)

# Add --diff flag for check mode
if [ "$CHECK_ONLY" = true ]; then
    YAPF_FLAGS+=('--diff')
    BLACK_FLAGS=('--check' '--diff')
    ISORT_FLAGS=('--check-only' '--diff')
else
    YAPF_FLAGS+=('--in-place')
    BLACK_FLAGS=()
    ISORT_FLAGS=()
fi

YAPF_EXCLUDES=(
    '--exclude' 'build/**'
    '--exclude' '.pytest_cache/**'
    '--exclude' 'robodm.egg-info/**'
    '--exclude' '__pycache__/**'
)

ISORT_EXCLUDES=(
    '--sg' 'build/**'
    '--sg' '.pytest_cache/**'
    '--sg' 'robodm.egg-info/**'
    '--sg' '__pycache__/**'
)

PYLINT_FLAGS=(
    '--disable=C0103,C0114,C0115,C0116'  # Disable some overly strict checks
)

# Track if any formatting issues were found
FORMAT_ISSUES=false

# Format specified files
format() {
    if [ "$CHECK_ONLY" = true ]; then
        if ! yapf "${YAPF_FLAGS[@]}" "$@" | grep -q .; then
            return 0
        else
            echo "YAPF formatting issues found"
            FORMAT_ISSUES=true
            return 1
        fi
    else
        yapf "${YAPF_FLAGS[@]}" "$@"
    fi
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
    MERGEBASE="$(git merge-base origin/main HEAD 2>/dev/null || git merge-base origin/master HEAD 2>/dev/null || echo HEAD~1)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        local files
        files=$(git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi')
        if [ -n "$files" ]; then
            echo "$files" | tr '\n' '\0' | xargs -P 5 -0 \
                yapf "${YAPF_EXCLUDES[@]}" "${YAPF_FLAGS[@]}"
        fi
    fi
}

# Format all files
format_all() {
    if [ "$CHECK_ONLY" = true ]; then
        echo "Checking YAPF formatting..."
        if ! yapf "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" robodm tests examples | grep -q .; then
            echo "✓ YAPF: No formatting issues"
        else
            echo "✗ YAPF: Formatting issues found"
            FORMAT_ISSUES=true
        fi
    else
        yapf "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" robodm tests examples
    fi
}

echo 'robodm Black formatting:'
if [ "$CHECK_ONLY" = true ]; then
    echo "Checking Black formatting..."
    if black "${BLACK_FLAGS[@]}" robodm tests examples; then
        echo "✓ Black: No formatting issues"
    else
        echo "✗ Black: Formatting issues found"
        FORMAT_ISSUES=true
    fi
else
    black "${BLACK_FLAGS[@]}" robodm tests examples
fi

## This flag formats individual files. --files *must* be the first command line
## arg to use this option.
if [[ "$1" == '--files' ]]; then
   format "${@:2}"
   # If `--all` is passed, then any further arguments are ignored and the
   # entire python directory is formatted.
elif [[ "$RUN_ALL" == true ]]; then
   format_all
else
   # Format only the files that changed in last commit.
   format_changed
fi
echo 'robodm yapf: Done'

echo 'robodm isort:'
if [ "$CHECK_ONLY" = true ]; then
    echo "Checking isort formatting..."
    if isort "${ISORT_FLAGS[@]}" robodm tests examples "${ISORT_EXCLUDES[@]}"; then
        echo "✓ isort: No formatting issues"
    else
        echo "✗ isort: Formatting issues found"
        FORMAT_ISSUES=true
    fi
else
    isort "${ISORT_FLAGS[@]}" robodm tests examples "${ISORT_EXCLUDES[@]}"
fi

# Run mypy
echo 'robodm mypy:'
# Check if there are any Python files to check
if find robodm -name "*.py" | head -1 | grep -q .; then
    if mypy robodm --ignore-missing-imports --check-untyped-defs; then
        echo "✓ MyPy: No type issues"
    else
        echo "✗ MyPy: Type issues found"
        if [ "$CHECK_ONLY" = true ]; then
            FORMAT_ISSUES=true
        fi
    fi
else
    echo "No Python files found in robodm/"
fi

# Run Pylint
echo 'robodm Pylint:'
if [[ "$1" == '--files' ]]; then
    # If --files is passed, filter to files within robodm/ and pass to pylint.
    if pylint "${PYLINT_FLAGS[@]}" "${@:2}"; then
        echo "✓ Pylint: No issues"
    else
        echo "✗ Pylint: Issues found"
        if [ "$CHECK_ONLY" = true ]; then
            FORMAT_ISSUES=true
        fi
    fi
elif [[ "$RUN_ALL" == true ]]; then
    # Pylint entire robodm directory.
    if find robodm -name "*.py" | head -1 | grep -q .; then
        if pylint "${PYLINT_FLAGS[@]}" robodm; then
            echo "✓ Pylint: No issues"
        else
            echo "✗ Pylint: Issues found"
            if [ "$CHECK_ONLY" = true ]; then
                FORMAT_ISSUES=true
            fi
        fi
    else
        echo "No Python files found in robodm/"
    fi
else
    # Pylint only files in robodm/ that have changed in last commit.
    MERGEBASE="$(git merge-base origin/main HEAD 2>/dev/null || git merge-base origin/master HEAD 2>/dev/null || echo HEAD~1)"
    changed_files=$(git diff --name-only --diff-filter=ACM "$MERGEBASE" -- 'robodm/*.py' 'robodm/**/*.py')
    if [[ -n "$changed_files" ]]; then
        if echo "$changed_files" | tr '\n' '\0' | xargs -0 pylint "${PYLINT_FLAGS[@]}"; then
            echo "✓ Pylint: No issues"
        else
            echo "✗ Pylint: Issues found"
            if [ "$CHECK_ONLY" = true ]; then
                FORMAT_ISSUES=true
            fi
        fi
    else
        echo 'Pylint skipped: no files changed in robodm/.'
    fi
fi

# Final status check
if [ "$CHECK_ONLY" = true ]; then
    if [ "$FORMAT_ISSUES" = true ]; then
        echo ""
        echo "❌ Code formatting/quality issues detected!"
        echo "Please run 'bash format.sh --all' to fix formatting issues."
        exit 1
    else
        echo ""
        echo "✅ All code formatting and quality checks passed!"
        exit 0
    fi
fi

if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted files. Please review and stage the changes.'
    echo 'Changes not staged for commit:'
    echo
    git --no-pager diff --name-only
    exit 1
fi

echo 'robodm formatting complete!'