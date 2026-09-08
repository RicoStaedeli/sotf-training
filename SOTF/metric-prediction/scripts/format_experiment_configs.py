#!/usr/bin/env python3
"""
Reformat all forecasting experiment YAML files so that:
  - top-level keys (hydra, dry_run, batch_name, ...) are separated by blank lines
  - every run entry in `runs:` is preceded by a blank line
  - list items under `runs:` are indented (  - experiment_name:)
"""

import yaml
from pathlib import Path

FORECASTING = Path(
    "/Users/ricostadeli/Documents/HSG/thesis/ms-thesis"
    "/metric-prediction/src/config/configs/experiments/forecasting"
)


class IndentedDumper(yaml.Dumper):
    """Dump YAML with indented list items (  - item instead of - item)."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow=flow, indentless=False)


def _dump(data):
    return yaml.dump(
        data,
        Dumper=IndentedDumper,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    )



def reformat(path: Path) -> None:
    with open(path) as f:
        raw = f.read()

    # Preserve the header comment lines (everything before the first non-comment line)
    header_lines = []
    for line in raw.splitlines():
        if line.startswith("#") or line.strip() == "":
            header_lines.append(line)
        else:
            break
    # Trim trailing blank lines from header
    while header_lines and header_lines[-1].strip() == "":
        header_lines.pop()
    header = "\n".join(header_lines)

    data = yaml.safe_load(raw)
    if data is None:
        return

    # ── top-level structure ──────────────────────────────────────────────────
    top_keys = ["hydra", "dry_run"]
    meta_keys = ["batch_name", "output_root", "batch_root_subdir", "evaluation"]
    body_keys = ["common_overrides"]

    top = {k: data[k] for k in top_keys if k in data}
    meta = {k: data[k] for k in meta_keys if k in data}
    body = {k: data[k] for k in body_keys if k in data}
    runs = data.get("runs") or []

    # Convert all short scalar lists to flow style (inline)
    body = _walk_flow_lists(body)
    runs = _walk_flow_lists(runs)

    sections = []
    if top:
        sections.append(_dump(top).rstrip())
    if meta:
        sections.append(_dump(meta).rstrip())
    if body:
        sections.append(_dump(body).rstrip())

    # Render runs with a blank line before each entry
    if runs:
        runs_lines = ["runs:"]
        for run in runs:
            run_yaml = _dump([run]).rstrip()
            # _dump([run]) gives "- experiment_name: ...\n  overrides: ..."
            # indent by 2 spaces
            indented = "\n".join("  " + l if l else "" for l in run_yaml.splitlines())
            runs_lines.append("")  # blank line before each run
            runs_lines.append(indented)
        sections.append("\n".join(runs_lines))

    body_text = "\n\n".join(sections)
    content = header + "\n\n" + body_text + "\n"

    path.write_text(content)
    print(f"reformatted {path.relative_to(FORECASTING.parent.parent.parent.parent.parent.parent)}")


class _FlowList(list):
    pass


def _flow_list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _is_scalar(v):
    return isinstance(v, (int, float, bool, str))


def _walk_flow_lists(obj):
    """Recursively convert short all-scalar lists to _FlowList for inline rendering."""
    if isinstance(obj, dict):
        return {k: _walk_flow_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and all(_is_scalar(x) for x in obj):
            return _FlowList(obj)
        return [_walk_flow_lists(x) for x in obj]
    return obj


IndentedDumper.add_representer(_FlowList, _flow_list_representer)


def main():
    files = sorted(FORECASTING.rglob("*.yaml"))
    # skip archive
    files = [f for f in files if "archive" not in f.parts]
    print(f"Reformatting {len(files)} files...")
    for f in files:
        reformat(f)
    print("Done.")


if __name__ == "__main__":
    main()
