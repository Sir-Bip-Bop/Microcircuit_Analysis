#!/bin/bash
for d in 12.12 16.19; do
  echo "=== $d start $(date) ==="
  rm -rf "packet_full/$d"
  python3 -u packet_sim.py --drives "$d" \
      --n-trials 150 --amplitudes 0 0.025 0.05 0.1 \
      --out ./packet_full 2>&1 | tail -5
  if [ -f "packet_full/$d/schedule.json" ]; then
    echo "=== $d OK $(date) ==="
  else
    echo "=== $d FAILED, continuing ==="
  fi
done
echo "=== all done $(date) ==="
