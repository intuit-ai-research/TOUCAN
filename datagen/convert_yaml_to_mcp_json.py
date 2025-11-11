#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception as e:
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from e


def sanitize_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip()).strip("-")
    return safe.lower()


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


def build_tools(tool_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for tool in tool_entries or []:
        name = tool.get("tool_name") or tool.get("name") or "Unknown Tool"
        description = tool.get("description") or "No description available"
        parameters = tool.get("parameters") or {}
        input_schema = to_json_schema_from_parameters(parameters)
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


def map_yaml_to_json(server_key: str, server_data: Dict[str, Any], source_path: str) -> Dict[str, Any]:
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

    tools = build_tools(tools_yaml)
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
        mapped = map_yaml_to_json(server_key, server_data, input_yaml)
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
    return p.parse_args()


def main() -> None:
    args = get_args()
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


