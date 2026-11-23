#!/bin/bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE="$ROOT/code"
MASTER="$ROOT/sequence.log"

run_one() {
  local seed="$1"
  local name="$2"
  local epochs="$3"
  local out="$ROOT/outputs/$name"
  local log="$ROOT/${name}.log"

  mkdir -p "$out"
  {
    echo "=== $name: dimensionally corrected residual expressions ==="
    sed -n '299,361p' "$CODE/utils.py"
    echo "=== $name: init_seed=$seed, reference_collocation_seed=2601, epochs=$epochs, from scratch ==="
    echo "=== fixed scales: ion_cont=0.6100892858913114 energy=0.965239365788521 ion_mom=0.7936910647527818 ==="
  } > "$log"

  /usr/bin/env \
    SEED_OVERRIDE="$seed" \
    EPOCHS_OVERRIDE="$epochs" \
    LBFGS_STEPS_OVERRIDE=0 \
    CKPT_IN_OVERRIDE= \
    OUTPUT_OVERRIDE="$out" \
    CKPT_OUT_OVERRIDE="$out/${name}_checkpoint.pt" \
    SUMMARY_OVERRIDE="$out/VALIDATION_${name}.md" \
    OPTIMIZER_OVERRIDE=ADAM \
    /usr/bin/python3 -u "$CODE/mainScript.py" >> "$log" 2>&1
  local status=$?
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $name exit=$status" >> "$MASTER"
  return "$status"
}

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') sequence_start" > "$MASTER"

if ! run_one 7 dim_fix_5k 5000; then
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') stopped: dim_fix_5k process failed" >> "$MASTER"
  exec /bin/sleep 31536000
fi
if ! /usr/bin/python3 "$ROOT/check_dim_fix_5k.py" >> "$MASTER" 2>&1; then
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') stopped: dim_fix_5k gate failed" >> "$MASTER"
  exec /bin/sleep 31536000
fi

for spec in "7 p5_reference_v3" "1234 seed_sweep_1234" "2601 seed_sweep_2601"; do
  set -- $spec
  run_one "$1" "$2" 15000
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') stopped_after_failure=$2" >> "$MASTER"
    exec /bin/sleep 31536000
  fi
done

/usr/bin/python3 "$ROOT/analyze_reference.py" >> "$MASTER" 2>&1
analysis_status=$?
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') analysis exit=$analysis_status" >> "$MASTER"
touch "$ROOT/SEQUENCE_COMPLETE"
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') sequence_complete" >> "$MASTER"
exec /bin/sleep 31536000
