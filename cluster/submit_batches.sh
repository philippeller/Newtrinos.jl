#!/bin/bash

CHUNK_SIZE=25
TOTAL_POINTS=961
SUBMIT_SCRIPT="cluster/submit.sh"

previous_job_id=""

for start in $(seq 1 "${CHUNK_SIZE}" "${TOTAL_POINTS}"); do
  end=$((start + CHUNK_SIZE - 1))
  if [ "$end" -gt "$TOTAL_POINTS" ]; then
    end=$TOTAL_POINTS
  fi

  if [ -n "$previous_job_id" ]; then
    echo "Waiting for previous job ${previous_job_id} to finish..."
    while squeue -h -j "$previous_job_id" >/dev/null 2>&1; do
      sleep 30
    done
  fi

  echo "Submitting array chunk ${start}-${end}..."
  output=$(sbatch --array=${start}-${end} "${SUBMIT_SCRIPT}" 2>&1)
  if [ $? -ne 0 ]; then
    echo "Submission failed:"
    echo "$output"
    exit 1
  fi

  previous_job_id=$(echo "$output" | awk '{print $4}')
  echo "Submitted batch job ${previous_job_id}."
done
