#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple, Optional
import yaml


def standardize(name: str) -> str:
    if name is None:
        return ""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def load_tool_specs_by_category(root_dir: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Load all tool JSONs under root_dir/<category>/*.json
    Returns mapping: (category_name, standardized_tool_name) -> tool_json
    """
    mapping: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            continue
        for file in os.listdir(category_path):
            if not file.endswith(".json"):
                continue
            fp = os.path.join(category_path, file)
            with open(fp, "r") as f:
                data = json.load(f)
            # std_from_file = data.get("standardized_name")
            # tool_name = data.get("tool_name") or data.get("name") or os.path.splitext(file)[0]
            # std_tool = standardize(std_from_file or tool_name)
            mapping[(category, data.get("tool_name"))] = data
    return mapping


def normalize_param_list(params: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(params, list):
        return out
    for p in params:
        if not isinstance(p, dict):
            continue
        out.append(
            {
                "name": p.get("name", ""),
                "type": str(p.get("type", "")).lower() if p.get("type") is not None else "",
                "description": p.get("description", ""),
                "default": p.get("default", ""),
            }
        )
    return out


def parse_template_response(schema_field: Any) -> Any:
    """
    Try to convert a 'schema' field from the tool json endpoint to a template_response.
    - If dict: return as-is.
    - If string: attempt to json.loads, else {}.
    - Else: {}.
    """
    if isinstance(schema_field, dict):
        return schema_field
    if isinstance(schema_field, str) and schema_field.strip():
        try:
            return json.loads(schema_field)
        except Exception:
            return {}
    return {}


def build_api_list(item: Dict[str, Any], specs: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    api_entries: List[Dict[str, Any]] = []
    for mcp_server in item["metadata"]["mcp_servers"]:
        server_name = mcp_server["server_name"]
        category_name = mcp_server["labels"]["primary_label"]
        tool_json = specs[(category_name, server_name)]
        for api in tool_json["api_list"]:
            api_name = api["name"]
            api_desc = api["description"]
            method = api["method"]
            req_params = normalize_param_list(api["required_parameters"])
            opt_params = normalize_param_list(api["optional_parameters"])
            # template_response = api["schema"]
            api_entries.append(
                {
                    "category_name": category_name,
                    "tool_name": server_name,
                    "api_name": api_name,
                    "api_description": api_desc,
                    "required_parameters": req_params,
                    "optional_parameters": opt_params,
                    "method": method,
                    # "template_response": template_response,
                }
            )
    return api_entries


def extract_category_from_item(item: Dict[str, Any]) -> Optional[str]:
    try:
        mcp_servers = item["metadata"]["mcp_servers"]
        if not mcp_servers:
            return None
        server_info = mcp_servers[0].get("server_info", {})
        categories = server_info.get("categories") or []
        if categories and isinstance(categories, list):
            return categories[0]
        # Fallback: parse from original_file path
        original_file = mcp_servers[0].get("original_file", "")
        parts = original_file.split(os.sep)
        # heuristic: category likely appears as a directory near the end, e.g., .../StableToolBench_D1/<Category>/<file>.yaml
        for i, part in enumerate(parts):
            if part == "StableToolBench_D1" and i + 1 < len(parts):
                return parts[i + 1]
        return None
    except Exception:
        return None


def extract_tool_name_from_item(item: Dict[str, Any]) -> Optional[str]:
    # Prefer explicit 'target_tools' if present
    t = item.get("target_tools")
    if isinstance(t, str) and t.strip():
        return t.strip()
    # Fallback: first tool name from remote_server_response
    try:
        mcp_servers = item["metadata"]["mcp_servers"]
        if not mcp_servers:
            return None
        tool_names = mcp_servers[0]["remote_server_response"].get("tool_names") or []
        if tool_names:
            return tool_names[0]
    except Exception:
        pass
    return None


def convert_preview_to_g1(preview_file: str, tools_root: str, output_dir: str) -> None:
    preview = []
    with open(preview_file, "r") as f:
        for lines in f:
            item = json.loads(lines)
            preview.append(item)
    specs = load_tool_specs_by_category(tools_root)

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(preview):
        if not isinstance(item, dict):
            continue
        question = item["metadata"]["question"]
        category_name = extract_category_from_item(item)

        api_list = build_api_list(item, specs)
        relevant_apis = build_relevant_apis(item)
        g1_entry = {
            "api_list": api_list,
            "query": question,
            "relevant APIs": relevant_apis,
            "query_id": idx,
        }
        results.append(g1_entry)

    # split the results to 200 queries per file
    for i in range(0, len(results), 200):
        chunk = results[i:i+200]
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"generated_{i//200+1}.json"), "w") as out:
            json.dump(chunk, out, indent=4, ensure_ascii=False)


def build_relevant_apis(item: Dict[str, Any]) -> List[Tuple[str, str]]:    
    # metadata --> mcp_servers[0] --> remote_server_response --> tools[0] --> name
    relevant_apis = []
    for server in item["metadata"]["mcp_servers"]:
        # read yaml from original_file
        with open(server["original_file"], "r") as f:
            yaml_file = yaml.safe_load(f)
        api_name = list(yaml_file["mcp_servers"].keys())[0]
        remote_server_response = server["remote_server_response"]
        tool_names = [tool["name"] for tool in remote_server_response["tools"]]
        relevant_apis.extend([[api_name, tool_name] for tool_name in tool_names])
    return relevant_apis


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert preview ToolUse JSON into G1_category-like JSON.")
    p.add_argument("--preview_file", required=True, help="Path to preview JSON (e.g., preview_ToolUse_..._sanitized.json)")
    p.add_argument("--tools_root_dir", required=True, help="Root directory containing category subfolders with tool JSON specs")
    p.add_argument("--output_dir", required=True, help="Path to write the G1_category-like JSON")
    return p.parse_args()


def main() -> None:
    args = get_args()
    convert_preview_to_g1(args.preview_file, args.tools_root_dir, args.output_dir)
    print(f"Wrote: {args.output_dir}")


if __name__ == "__main__":
    main()


