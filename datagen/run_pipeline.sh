#!/usr/bin/env bash
set -euo pipefail

ORIG_PWD="$(pwd)"
ORIG_ARGS=("$@")

EXEC_LOG=""

print_cmd() {
  local line="+"
  local quoted=""
  for arg in "$@"; do
    printf -v quoted "%q" "$arg"
    line+=" $quoted"
  done
  echo "$line"
  if [[ -n "${EXEC_LOG:-}" ]]; then
    echo "$line" >> "$EXEC_LOG"
  fi
}

run_cmd() {
  print_cmd "$@"
  "$@"
}

run_pipeline_cmd() {
  # Print a pipeline command as a single, copy/paste-able line and then execute it.
  # Usage: run_pipeline_cmd "<command string>"
  local cmd_str="${1:?cmd string required}"
  local line="+ $cmd_str"
  echo "$line"
  if [[ -n "${EXEC_LOG:-}" ]]; then
    echo "$line" >> "$EXEC_LOG"
  fi
  bash -lc "$cmd_str"
}

usage() {
  cat <<'EOF'
Usage:
  ./run_pipeline.sh --input_dir <DIR> [options]

Required:
  --input_dir DIR             Input dir for stage0 (same meaning as in dvc.yaml stage0)
  --tools_root_dir DIR        Directory path to tools API definitions

Stage0:
  --mcp_servers_dir DIR       Where to write MCP server JSONs (default: ../mcp_servers)
  --cache_dir DIR             Cache dir for convert_yaml_to_mcp_json.py (default: ./tool_response_cache)

Stage1-1 (question generation):
  --num_tools N               (default: 2)
  --sampling_strategy STR     random|uniform|power_law|featured (default: uniform)
  --samples_per_server N      (default: 10)
  --mode STR                  single_server|multi_server (default: single_server)
  --output_folder DIR         Output root for stage1-1 (default: ../data)
  --timestamp INT             Timestamp/seed for stage1-1 (default: now)

Stage1-2 (completion):
  --model_name STR            Model name passed to completion scripts (default: gpt-4.1-2025-04-14)
  --engine STR                vllm_api|vllm|hf|together_api|openai|openrouter_api (default: openai)
  --start_vllm_service BOOL   true|false (default: false)

OpenAI:
  --openai_api_key STR        API key for OpenAI models (or set OPENAI_API_KEY env var)

Notes:
  - Always begins by removing ../mcp_servers (or --mcp_servers_dir).
  - Parses the LAST stdout line from stage1-1: "Output directory: ..._prepared.jsonl"
  - Auto-finds *_prepared.jsonl, *_results.jsonl, and *_4prepared.jsonl as requested.
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

INPUT_DIR=""
MCP_SERVERS_DIR="../mcp_servers"

NUM_TOOLS="2"
SAMPLING_STRATEGY="uniform"
SAMPLES_PER_SERVER="10"
MODE="single_server"
OUTPUT_FOLDER="../data"
TIMESTAMP="$(date +%s)"

ENGINE="openai"
START_VLLM_SERVICE="false"

OPENAI_API_KEY_ARG=""
MODEL_NAME="gpt-4.1-2025-04-14"

TOOLS_ROOT_DIR=""
CACHE_DIR="./tool_response_cache"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_dir) INPUT_DIR="${2:-}"; shift 2 ;;
    --mcp_servers_dir) MCP_SERVERS_DIR="${2:-}"; shift 2 ;;

    --num_tools) NUM_TOOLS="${2:-}"; shift 2 ;;
    --sampling_strategy) SAMPLING_STRATEGY="${2:-}"; shift 2 ;;
    --samples_per_server) SAMPLES_PER_SERVER="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --output_folder) OUTPUT_FOLDER="${2:-}"; shift 2 ;;
    --timestamp) TIMESTAMP="${2:-}"; shift 2 ;;

    --engine) ENGINE="${2:-}"; shift 2 ;;
    --start_vllm_service) START_VLLM_SERVICE="${2:-}"; shift 2 ;;

    --openai_api_key) OPENAI_API_KEY_ARG="${2:-}"; shift 2 ;;
    --model_name) MODEL_NAME="${2:-}"; shift 2 ;;

    --tools_root_dir) TOOLS_ROOT_DIR="${2:-}"; shift 2 ;;
    --cache_dir) CACHE_DIR="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1 (use --help)" ;;
  esac
done

[[ -n "$INPUT_DIR" ]] || { usage; die "--input_dir is required"; }
[[ -n "$TOOLS_ROOT_DIR" ]] || { usage; die "--tools_root_dir is required"; }

# Resolve OpenAI API key: CLI arg takes precedence over env var
if [[ -n "$OPENAI_API_KEY_ARG" ]]; then
  export OPENAI_API_KEY="$OPENAI_API_KEY_ARG"
elif [[ -z "${OPENAI_API_KEY:-}" ]]; then
  die "OpenAI API key required. Pass --openai_api_key or set OPENAI_API_KEY env var."
fi


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

command -v python >/dev/null 2>&1 || die "python not found on PATH"
command -v jq >/dev/null 2>&1 || die "jq not found on PATH (needed for final combine step)"

MCP_SERVERS_DIR_ABS="$(python - <<PY
import os,sys
print(os.path.abspath(os.path.join("$SCRIPT_DIR", "$MCP_SERVERS_DIR")))
PY
)"

echo "==> [0] Cleaning MCP servers dir: $MCP_SERVERS_DIR_ABS"
run_cmd rm -rf "$MCP_SERVERS_DIR_ABS"
run_cmd mkdir -p "$MCP_SERVERS_DIR_ABS"

echo "==> [0] Stage0: convert_yaml_to_mcp_json.py"
run_cmd python convert_yaml_to_mcp_json.py --input_dir "$INPUT_DIR" --output_dir "$MCP_SERVERS_DIR_ABS" --cache_dir "$CACHE_DIR"

echo "==> [1-1] Stage1-1: step1.1_gen_questions.py"
stage1_1_log="$(mktemp -t stage1_1.XXXXXX.log)"
run_pipeline_cmd "python step1.1_gen_questions.py \
  --num_tools $(printf %q "$NUM_TOOLS") \
  --sampling_strategy $(printf %q "$SAMPLING_STRATEGY") \
  --samples_per_server $(printf %q "$SAMPLES_PER_SERVER") \
  --mode $(printf %q "$MODE") \
  --output_folder $(printf %q "$OUTPUT_FOLDER") \
  --timestamp $(printf %q "$TIMESTAMP") \
  | tee $(printf %q "$stage1_1_log")"

last_line="$(tail -n 1 "$stage1_1_log" || true)"
run_cmd rm -f "$stage1_1_log"

[[ "$last_line" == Output\ directory:* ]] || die "Stage1-1 did not end with expected line. Got: $last_line"
prepared_from_stdout="${last_line#Output directory: }"
prepared_from_stdout="$(echo "$prepared_from_stdout" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
[[ -n "$prepared_from_stdout" ]] || die "Parsed empty prepared path from: $last_line"

target_dir="$(dirname "$prepared_from_stdout")"
target_dir_abs="$(python - <<PY
import os
print(os.path.abspath("$target_dir"))
PY
)"

echo "==> Target folder determined from stage1-1 stdout: $target_dir_abs"

EXEC_LOG="$target_dir_abs/execution_log.txt"
{
  echo "# Execution log (commands printed by run_pipeline.sh after target dir was determined)"
  echo "# Generated: $(date '+%Y-%m-%dT%H:%M:%S')"
  echo "# Target dir: $target_dir_abs"
  echo
} >> "$EXEC_LOG"

echo "==> Writing reproducibility command: $target_dir_abs/command.txt"
script_abs="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
{
  echo "# Reproducibility command"
  echo "# Generated: $(date '+%Y-%m-%dT%H:%M:%S')"
  echo "# Original working dir: $ORIG_PWD"
  printf "%q " "$script_abs" "${ORIG_ARGS[@]}"
  echo
} > "$target_dir_abs/command.txt"

echo "==> [1-2] Stage1-2: find *_prepared.jsonl in target folder and run completion"
prepared_files=()
while IFS= read -r f; do prepared_files+=("$f"); done < <(find "$target_dir_abs" -maxdepth 1 -type f -name "*_prepared.jsonl" | sort)
[[ ${#prepared_files[@]} -gt 0 ]] || die "No *_prepared.jsonl found in $target_dir_abs"

for pf in "${prepared_files[@]}"; do
  echo "----> Completing: $pf"
  run_cmd bash step1.2_completion.sh "$pf" "$MODEL_NAME" "$ENGINE" "1.2" "$START_VLLM_SERVICE"
done

echo "==> [1-3] Stage1-3: find *_results.jsonl in target folder and process"
results_files=()
while IFS= read -r f; do results_files+=("$f"); done < <(find "$target_dir_abs" -maxdepth 1 -type f -name "*_results.jsonl" | sort)
[[ ${#results_files[@]} -gt 0 ]] || die "No *_results.jsonl found in $target_dir_abs (did stage1-2 run successfully?)"

for rf in "${results_files[@]}"; do
  echo "----> Processing: $rf"
  run_cmd python step1.3_process_completion.py --input_file "$rf"
done

echo "==> [convert2query] Find *_4prepared.jsonl under processed/ and convert to queries/"
processed_dir="$target_dir_abs/processed"
[[ -d "$processed_dir" ]] || die "Missing processed dir: $processed_dir (did stage1-3 run successfully?)"

preview_files=()
while IFS= read -r f; do preview_files+=("$f"); done < <(find "$processed_dir" -maxdepth 1 -type f -name "ToolUse_*_4prepared.jsonl" | sort)
[[ ${#preview_files[@]} -gt 0 ]] || die "No ToolUse_*_4prepared.jsonl found in $processed_dir"

if [[ ${#preview_files[@]} -gt 1 ]]; then
  echo "WARNING: multiple ToolUse_*_4prepared.jsonl found; using the newest by mtime." >&2
fi
preview_file="$(ls -t "${preview_files[@]}" | head -n 1)"

queries_dir="$target_dir_abs/queries"
run_cmd mkdir -p "$queries_dir"

run_cmd python convert_preview_to_g1.py \
  --preview_file "$preview_file" \
  --tools_root_dir "$TOOLS_ROOT_DIR" \
  --output_dir "$queries_dir"

echo "==> [combine] jq -s 'add' queries/*.json > combined_queries.json"
shopt -s nullglob
jsons=( "$queries_dir"/*.json )
shopt -u nullglob
[[ ${#jsons[@]} -gt 0 ]] || die "No .json files found in $queries_dir to combine"

print_cmd jq -s 'add' "$queries_dir"/*.json ">" "$target_dir_abs/combined_queries.json"
jq -s 'add' "$queries_dir"/*.json > "$target_dir_abs/combined_queries.json"
echo "Wrote combined queries: $target_dir_abs/combined_queries.json"


