for m in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3-0.6B google/gemma-3-1b-it HuggingFaceTB/SmolLM2-1.7B-Instruct microsoft/Phi-3.5-mini-instruct; do
  echo "=== $m"; UTIL=${UTIL:-0.4} .venv/bin/python tests/check_plugin_exact.py 1.5 $m 2>&1 | grep -vE "^\s+File|^\s+(next|raise|wait_for)|\^\^" | tail -8
done
echo ALLDONE
