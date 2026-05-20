import json
import re
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]

method_dir = repo_root / "docs/methods"
dataset_dir = repo_root / "docs/datasets"
commercial_file = repo_root / "docs/commercial.md"
readme_file = repo_root / "README.md"

def count_table_rows(file):
    total = 0
    for line in file.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells:
            continue
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        if cells[0] in {"Model", "Dataset", "Resource", "Category"}:
            continue
        total += 1
    return total

def count_models():
    total = 0
    for f in method_dir.glob("*.md"):
        total += count_table_rows(f)
    return total

def count_datasets():
    total = 0
    for f in dataset_dir.glob("*.md"):
        total += count_table_rows(f)
    return total

def count_commercial():
    total = 0
    for line in commercial_file.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 4 or cells[0] in {"Family", "---"}:
            continue
        for cell in (cells[2], cells[3]):
            items = [item.strip() for item in cell.split("<br>")]
            total += sum(1 for item in items if item and item != "—")
    return total

def update_readme(stats):
    text = readme_file.read_text()
    replacements = {
        "Biomedical MLLM Models": stats["models"],
        "Biomedical Datasets": stats["datasets"],
        "Commercial Models": stats["commercial_models"],
    }
    updated = text
    for label, value in replacements.items():
        updated, count = re.subn(
            rf"(\| {re.escape(label)} \| )\d+(\s*\|)",
            rf"\g<1>{value}\g<2>",
            updated,
            count=1,
        )
        if count not in {0, 1}:
            raise RuntimeError(f"Could not update {label} in README.md")
    if updated != text:
        readme_file.write_text(updated)

stats = {
    "models": count_models(),
    "datasets": count_datasets(),
    "commercial_models": count_commercial()
}

output = repo_root / "docs/stats.json"

with open(output, "w") as f:
    json.dump(stats, f, indent=2)

update_readme(stats)

print("Updated stats:", stats)
