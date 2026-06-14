#!/bin/bash
# Test a range of points locally to identify where failures start
# Usage: ./test_range.sh [START] [END]

START=${1:-1}
END=${2:-10}

echo "Testing points $START to $END locally..."

for i in $(seq $START $END); do
    echo "=========================================="
    echo "Point $i:"
    
    # Run with memory limit to catch OOM
    # Using ulimit to track memory usage
    (ulimit -v 2000000  # ~2GB virtual memory limit
     timeout 60 julia --project=@. single_point.jl $i $END) 2>&1 | tail -5
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ SUCCESS"
    elif [ $EXIT_CODE -eq 137 ]; then
        echo "✗ OOM KILL (exit 137)"
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "✗ TIMEOUT (exit 124)"
    else
        echo "✗ FAILED (exit $EXIT_CODE)"
    fi
    
    echo ""
done

echo "Test complete"
