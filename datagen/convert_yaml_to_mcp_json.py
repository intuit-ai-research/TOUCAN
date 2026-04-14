#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
import ast
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception as e:
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from e


def sanitize_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip()).strip("-")
    return safe.lower()

class CacheIndex:
    """
    Index JSON cache files under a cache directory so we can attach cached examples
    to tool descriptions.

    We intentionally key by basename to support both:
    - flat caches: <cache_dir>/<api_name>.json
    - nested caches: <cache_dir>/<Category>/<ToolName>/<api_name>.json
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self._by_basename: Dict[str, List[str]] = {}
        if cache_dir and os.path.isdir(cache_dir):
            for dirpath, dirnames, filenames in os.walk(cache_dir):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for fn in filenames:
                    if fn.startswith(".") or not fn.lower().endswith(".json"):
                        continue
                    full = os.path.join(dirpath, fn)
                    self._by_basename.setdefault(fn, []).append(full)

    def find_best(self, wanted_basename: str, prefer_substrings: List[str]) -> Optional[str]:
        cands = self._by_basename.get(wanted_basename, [])
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        # Prefer paths that contain more of the preferred substrings (category/server/tool hints)
        scored: List[tuple[int, int, str]] = []
        for p in cands:
            path_l = p.lower()
            hits = 0
            for s in prefer_substrings:
                s = (s or "").strip().lower()
                if s and s in path_l:
                    hits += 1
            scored.append((hits, -len(p), p))
        scored.sort(reverse=True)
        return scored[0][2]

def _safe_str(obj: Any, max_chars: int) -> str:
    s = str(obj)
    if len(s) <= max_chars:
        return s
    # keep head+tail to preserve key info when it's a huge list/object
    head = s[: max_chars - 80]
    tail = s[-60:] if len(s) > 60 else ""
    return f"{head}…[truncated]…{tail}"

def _parse_cache_input_key(raw_key: Any) -> Optional[Dict[str, Any]]:
    """
    Cache keys often look like '{}', or "{'a': 1}", or "OrderedDict([...])".
    Best effort: parse into a dict for key-name extraction.
    """
    if raw_key is None:
        return None
    if isinstance(raw_key, dict):
        return raw_key
    if not isinstance(raw_key, str):
        return None
    s = raw_key.strip()
    if not s:
        return None
    try:
        val = ast.literal_eval(s)
        if isinstance(val, dict):
            return val
    except Exception:
        return None
    return None

def _cache_snippet_for_description(
    cache_path: str,
    max_total_chars: int = 1200,
    max_examples: int = 3,
    max_input_chars: int = 240,
    max_response_chars: int = 520,
) -> str:
    """
    Build a short, readable cache summary to append to a tool description.
    Focus on:
    - observed cached input keys (helps users know what parameters have been used)
    - a few truncated examples
    """
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return f"\n\nCached examples: [failed to load cache at {cache_path}: {e}]"

    if not isinstance(data, dict) or not data:
        return "\n\nCached examples: [no cached records found]"

    # Collect observed input keys
    observed_keys: List[str] = []
    examples: List[str] = []
    for i, (k, v) in enumerate(list(data.items())[: max_examples]):
        parsed = _parse_cache_input_key(k)
        if parsed:
            for kk in parsed.keys():
                sk = str(kk)
                if sk not in observed_keys:
                    observed_keys.append(sk)

        # v usually looks like {"error": "", "response": ...}
        resp = None
        err = None
        if isinstance(v, dict):
            err = v.get("error")
            resp = v.get("response")
        if isinstance(err, str) and len(err) > 1:
            # if error is not empty, skip the example
            continue
        input_s = _safe_str(parsed if parsed is not None else k, max_input_chars)
        out_s = _safe_str(resp if resp is not None else v, max_response_chars)
        examples.append(f"- input: {input_s}\n  output: {out_s}")

    keys_line = ""
    if observed_keys:
        keys_line = f"Observed cached input keys: {', '.join(observed_keys[:30])}."
        if len(observed_keys) > 30:
            keys_line += " …"

    body = "Cached examples (truncated):"
    if keys_line:
        body += f"\n{keys_line}"
    if examples:
        body += "\n" + "\n".join(examples)

    body = _safe_str(body, max_total_chars)
    return "\n\n" + body


def to_json_schema_from_parameters(parameters: Any) -> Dict[str, Any]:
    """
    Convert a 'parameters' dictionary (if available) into a JSON Schema.
    - If parameters is empty or not a dict, returns an empty object schema.
    - If parameters has keys, conservatively assume string-typed properties unless typed info exists.
    """
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#",
    }

    if isinstance(parameters, dict) and parameters:
        props: Dict[str, Any] = {}
        for key, val in parameters.items():
            # Best-effort typing inference
            inferred: Dict[str, Any] = {"type": "string"}
            if isinstance(val, dict):
                # If the source already looks schema-like, pass through limited fields
                t = val.get("type")
                d = val.get("description")
                if isinstance(t, str):
                    inferred["type"] = t
                if isinstance(d, str):
                    inferred["description"] = d
            schema_field: Dict[str, Any] = inferred
            props[str(key)] = schema_field
        schema["properties"] = props
    return schema


def _choose_cache_for_tool(
    tool: Dict[str, Any],
    cache_index: Optional[CacheIndex],
    server_name: Optional[str],
    category: Optional[str],
) -> Optional[str]:
    if not cache_index:
        return None

    # Heuristics: tools might have name/tool_name/api_name/operation_id
    candidates: List[str] = []
    for k in ("api_name", "operation_id", "name", "tool_name"):
        v = tool.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())
    # If the YAML author provides an explicit cache file reference, honor it.
    explicit = tool.get("cache_file") or tool.get("cache_path")
    if isinstance(explicit, str) and explicit.strip():
        p = explicit.strip()
        if not p.lower().endswith(".json"):
            p += ".json"
        # If relative, make it relative to cache_dir
        if not os.path.isabs(p):
            p = os.path.join(cache_index.cache_dir, p)
        if os.path.exists(p):
            return p

    prefer = [category or "", server_name or ""] + candidates
    # Try basename matches first (flat or nested)
    for cand in candidates:
        base = cand if cand.lower().endswith(".json") else f"{cand}.json"
        found = cache_index.find_best(base, prefer_substrings=prefer)
        if found:
            return found
    return None


def build_tools(
    tool_entries: List[Dict[str, Any]],
    cache_index: Optional[CacheIndex],
    server_name: Optional[str],
    category: Optional[str],
) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for tool in tool_entries or []:
        name = tool.get("tool_name") or tool.get("name") or "Unknown Tool"
        description = tool.get("description") or "No description available"
        parameters = tool.get("parameters") or {}
        input_schema = to_json_schema_from_parameters(parameters)

        cache_path = _choose_cache_for_tool(
            tool,
            cache_index=cache_index,
            server_name=server_name,
            category=category,
        )
        if cache_path:
            # Append cache info, but keep it bounded so we never blow up descriptions.
            description = f"{description}{_cache_snippet_for_description(cache_path)}"

        tools.append(
            {
                "name": name,
                "description": description,
                "input_schema": input_schema,
                "annotations": None,
            }
        )
    return tools


def derive_tags(category: Optional[str], platform: Optional[str]) -> List[str]:
    tags: List[str] = ["api", "mcp"]
    if category and isinstance(category, str):
        tags.append(category.lower())
    if platform and isinstance(platform, str):
        tags.append(platform.lower())
    return sorted(list(dict.fromkeys(tags)))


def map_yaml_to_json(
    server_key: str,
    server_data: Dict[str, Any],
    source_path: str,
    cache_index: Optional[CacheIndex],
) -> Dict[str, Any]:
    """
    Map one server entry from YAML into the JSON structure required by downstream.
    """
    # Core fields from YAML
    name = server_data.get("name") or server_key
    overview = server_data.get("description") or ""
    category = server_data.get("category") or ""

    # Attempt to read a platform hint from any tool's _metadata.platform
    platform: Optional[str] = None
    tools_yaml = server_data.get("tools") or []
    for t in tools_yaml:
        meta = t.get("_metadata") or {}
        if isinstance(meta, dict) and meta.get("platform"):
            platform = meta.get("platform")
            break

    tools = build_tools(
        tools_yaml,
        cache_index=cache_index,
        server_name=name,
        category=category,
    )
    tool_names = [t["name"] for t in tools]
    tools_count = len(tools)

    now = int(time.time())

    # Labels: keep minimal fields that the generator uses, plus primary_label from YAML category
    labels = {
        "primary_label": category or "Uncategorized",
        "secondary_labels": [],
        "custom_label": "",
        "is_connected": True,
        "is_remote_tool_valid": True,
        "featured_server": False,
    }

    # server_info_crawled: must include name, overview, remote_or_local == 'Remote'
    server_info_crawled: Dict[str, Any] = {
        "id": None,
        "name": name,
        "author": "",
        "overview": overview,
        "repository_url": "",
        "homepage": "",
        "remote_or_local": "Remote",
        "license": "",
        "usage_count": "Not available",
        "success_rate": "Not available",
        "tags": derive_tags(category, platform),
        "categories": [labels["primary_label"]] if labels["primary_label"] else [],
        "file_path": source_path,
        "tools_count": tools_count,
        "tools": tools,
        "python_sdk": "",
        "configuration_schema": "",
        "smithery_configuration_requirements": [],
        "python_sdk_config": "",
        "python_sdk_url": "",
    }

    # remote_server_response: must include non-empty tools
    remote_server_response: Dict[str, Any] = {
        "url": "",
        "is_success": True,
        "error": None,
        "tools": tools,
        "tool_count": tools_count,
        "tool_names": tool_names,
    }

    metadata: Dict[str, Any] = {
        "server_id": None,
        "server_name": name,
        "rank_by_usage": None,
        "usage_count": "Not available",
        "original_file": source_path,
        "mode": "smithery",
        "timestamp": now,
        "remote_server_response": remote_server_response,
        "server_info_crawled": server_info_crawled,
        "source_filename": os.path.basename(source_path),
        "processed_timestamp": now,
        "processing_mode": "smithery",
        "rank": None,
    }

    return {
        "labels": labels,
        "metadata": metadata,
    }


def convert_file(input_yaml: str, output_dir: str) -> List[str]:
    with open(input_yaml, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "mcp_servers" not in data or not isinstance(data["mcp_servers"], dict):
        raise ValueError("Input YAML must contain a top-level 'mcp_servers' mapping.")

    os.makedirs(output_dir, exist_ok=True)

    written: List[str] = []
    for server_key, server_data in data["mcp_servers"].items():
        if not isinstance(server_data, dict):
            continue
        # cache_dir is injected via a global created in main()
        mapped = map_yaml_to_json(server_key, server_data, input_yaml, cache_index=GLOBAL_CACHE_INDEX)
        server_name = mapped["metadata"]["server_name"] or server_key
        yaml_stem = os.path.splitext(os.path.basename(input_yaml))[0]
        base = f"tf_generated.{sanitize_name(yaml_stem)}.{sanitize_name(server_name)}_labeled.json"
        output_path = os.path.join(output_dir, base)
        with open(output_path, "w") as out:
            json.dump(mapped, out, indent=2)
        written.append(output_path)
    return written


def list_yaml_files(root_dir: str) -> List[str]:
    yaml_files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Do not descend into hidden directories (e.g., .ipynb_checkpoints, .git)
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            # Skip hidden files
            if fn.startswith("."):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in {".yaml", ".yml"}:
                yaml_files.append(os.path.join(dirpath, fn))
    return yaml_files


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert StableToolBench-like YAML into MCP labeled JSON.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to a single input YAML (e.g., text_to_speech_multiple_languages_voices.yaml)")
    group.add_argument("--input_dir", help="Directory to recursively search for YAML/YML files")
    p.add_argument("--output_dir", required=True, help="Directory to write JSON outputs")
    p.add_argument("--cache_dir", required=True, help="Directory to server cache")
    return p.parse_args()

GLOBAL_CACHE_INDEX: Optional[CacheIndex] = None

def main() -> None:
    args = get_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    global GLOBAL_CACHE_INDEX
    GLOBAL_CACHE_INDEX = CacheIndex(args.cache_dir)
    inputs: List[str] = []
    if args.input_dir:
        inputs = list_yaml_files(args.input_dir)
        if not inputs:
            raise SystemExit(f"No YAML files found under: {args.input_dir}")
    elif args.input:
        inputs = [args.input]
    else:
        raise SystemExit("Either --input or --input_dir must be provided.")

    outputs: List[str] = []
    for in_file in inputs:
        try:
            outputs.extend(convert_file(in_file, args.output_dir))
        except Exception as e:
            print(f"Failed to convert {in_file}: {e}")

    print("Wrote files:")
    for p in outputs:
        print(f"- {p}")


if __name__ == "__main__":
    main()


