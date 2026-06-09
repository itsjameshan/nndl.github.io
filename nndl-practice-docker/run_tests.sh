#!/usr/bin/env bash
# Verify the image: maintainer pytest suite + end-to-end execution of all 17
# pytorch/ notebooks. Writes a report to /workspace/report. Exit 0 only if
# pytest passes AND every notebook runs clean.
#
# Env:
#   NB_TIMEOUT   per-notebook cell timeout in seconds (default 3600)
#   ONLY         space-separated substring filter, e.g. ONLY="chap1 chap2"
set -u

ROOT=/workspace/nndl-practice
REPORT=/workspace/report
mkdir -p "$REPORT"
RESULTS="$REPORT/results.md"
TIMEOUT="${NB_TIMEOUT:-3600}"
ONLY="${ONLY:-}"
EXCLUDE="${EXCLUDE:-}"
SKIP_PYTEST="${SKIP_PYTEST:-}"

cd "$ROOT"

{
  echo "# nndl-practice test report"
  echo
  echo "- date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- arch: $(uname -m)"
  echo "- torch: $(python -c 'import torch;print(torch.__version__)' 2>/dev/null)"
  echo "- per-notebook timeout: ${TIMEOUT}s"
  echo
} > "$RESULTS"

# 1) pytest sanity suite ------------------------------------------------------
echo "== pytest =="
if [ -n "$SKIP_PYTEST" ]; then
  echo "(skipped via SKIP_PYTEST)" | tee "$REPORT/pytest.log"
  PYTEST_RC=0
else
  ( cd pytorch && python -m pytest tests -q ) 2>&1 | tee "$REPORT/pytest.log"
  PYTEST_RC=${PIPESTATUS[0]}
fi
{
  echo "## pytest"
  echo
  echo "exit code: ${PYTEST_RC} ($([ "$PYTEST_RC" -eq 0 ] && echo PASS || echo FAIL))"
  echo
} >> "$RESULTS"

# 2) execute every notebook ---------------------------------------------------
{
  echo "## notebooks"
  echo
  echo "| notebook | status | seconds |"
  echo "|---|---|---|"
} >> "$RESULTS"

FAIL=0
COUNT=0
# Sorted, no checkpoint copies. NUL-delimited to survive spaces in names.
while IFS= read -r -d '' nb; do
  name="${nb#pytorch/}"
  if [ -n "$ONLY" ]; then
    keep=0
    for f in $ONLY; do case "$name" in *"$f"*) keep=1;; esac; done
    [ "$keep" -eq 1 ] || continue
  fi
  if [ -n "$EXCLUDE" ]; then
    skip=0
    for f in $EXCLUDE; do case "$name" in *"$f"*) skip=1;; esac; done
    [ "$skip" -eq 0 ] || continue
  fi
  COUNT=$((COUNT + 1))
  logf="$REPORT/$(echo "$name" | tr '/ ' '__').log"
  # CPU_FAST: execute a shrunk throwaway copy beside the original (so the kernel's
  # cwd still resolves ../datasets), leaving the original notebook untouched.
  target="$nb"
  tmpnb=""
  if [ -n "$CPU_FAST" ]; then
    tmpnb="$(dirname "$nb")/._cpufast.ipynb"
    python /workspace/cpufast.py "$nb" "$tmpnb" && target="$tmpnb"
  fi
  start=$(date +%s)
  if jupyter nbconvert --to notebook --execute "$target" \
        --output /tmp/_executed.ipynb \
        --ExecutePreprocessor.timeout="$TIMEOUT" \
        --ExecutePreprocessor.kernel_name=python3 \
        > "$logf" 2>&1; then
    status="PASS"
  else
    status="FAIL"
    FAIL=$((FAIL + 1))
  fi
  [ -n "$tmpnb" ] && rm -f "$tmpnb"
  end=$(date +%s)
  echo "| ${name} | ${status} | $((end - start)) |" | tee -a "$RESULTS"
done < <(find pytorch -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' -print0 | sort -z)

{
  echo
  echo "notebooks run: ${COUNT}, failed: ${FAIL}"
  echo "pytest exit: ${PYTEST_RC}"
  echo
  if [ "$PYTEST_RC" -eq 0 ] && [ "$FAIL" -eq 0 ]; then
    echo "OVERALL: GREEN"
  else
    echo "OVERALL: RED"
  fi
} | tee -a "$RESULTS"

[ "$PYTEST_RC" -eq 0 ] && [ "$FAIL" -eq 0 ]
