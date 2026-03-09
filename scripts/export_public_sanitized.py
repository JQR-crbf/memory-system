#!/usr/bin/env python3
"""Create a GitHub-shareable sanitized copy of memory-system.

This script never mutates the source directory. It exports a curated copy that
keeps code and integration files while replacing private runtime data with
safe templates.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


INCLUDE_DIRS = ["src", "config", "integrations", "scripts"]
INCLUDE_FILES = ["README.md", ".env.example", ".gitkeep"]


ENV_EXAMPLE_CONTENT = """# Copy to .env and fill your own values.
NEO4J_AUTH=neo4j/CHANGE_ME
QDRANT_URL=http://qdrant:6333
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=CHANGE_ME
OLLAMA_BASE_URL=http://host.docker.internal:11434
GEN_MODEL=qwen3:8b
EMBED_MODEL=nomic-embed-text
CLASSIFIER_MODEL=qwen3:8b
CLASSIFIER_LLM_TEMPERATURE=0
CLASSIFIER_LLM_TOP_P=1
CLASSIFIER_LEARNING_ENABLED=true
CLASSIFIER_VECTOR_ENABLED=true
CLASSIFIER_LLM_FALLBACK_ENABLED=true
CLASSIFIER_VECTOR_THRESHOLD=0.78
CLASSIFIER_VECTOR_COLLECTION=classification_router
CLASSIFIER_VECTOR_TOP_K=3
CHAT_AUTOSAVE_ENABLED=true
CHAT_AUTOSAVE_MIN_SCORE=0.55
CHAT_AUTOSAVE_LLM_ENABLED=true
CHAT_AUTOSAVE_LLM_MODEL=qwen3:8b
CHAT_AUTOSAVE_LLM_TIMEOUT=20
CHAT_AUTOSAVE_REVIEW_BAND=0.2
L4_AUTO_REFLECT_ENABLED=false
L4_AUTO_REFLECT_INTERVAL_MINUTES=360
L4_AUTO_REFLECT_TYPE=weekly
L4_AUTO_REFLECT_FOCUS_AREAS=Core Outcomes,Growth Areas,Blind Spots,Next Actions
MEMORY_API_KEY=
"""


SANITIZED_PROFILE = """identity: '{"name": "匿名用户", "age": 0, "location": "未设置", "languages": ["中文"]}'
career: '{"current_role": "未设置", "company": "未设置", "years_of_experience": 0, "tech_stack": {"primary": [], "secondary": [], "learning": []}, "domains": []}'
personality:
  communication_style: 简洁
  values: []
  pet_peeves: []
  strengths: []
  weaknesses: []
writing_style:
  tone: 专业
  format_preference: 结构化
  length: 精简
  language: 中文
ai_instructions:
  always:
  - 用中文回答
  - 给出可执行建议
  never:
  - 泄露敏感信息
  - 编造事实
  context_rules:
  - 优先使用当前上下文
"""


SANITIZED_GOALS = """goals:
- id: goal-demo-001
  title: 构建可复用的个人记忆系统
  category: learning
  priority: high
  deadline: '2099-12-31'
  status: in_progress
  progress: 0
  sub_goals: []
  related_skills:
  - Memory System
  - RAG
  notes: 这是公开示例目标，无真实examples
"""


TEXT_SUFFIX_ALLOWLIST = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".sh",
    ".env",
    ".example",
    ".gitkeep",
    ".Dockerfile",
}


TEXT_REPLACEMENTS = [
    ("匿名用户", "匿名用户"),
    ("examples", "examples"),
    ("worklogs", "worklogs"),
    ("reports.sample.json", "reports.sample.json"),
    ("worklog.sample.md", "worklog.sample.md"),
    ("worklog.sample.md", "worklog.sample.md"),
]


def copy_curated_source(source: Path, target: Path) -> None:
    for folder in INCLUDE_DIRS:
        src = source / folder
        if src.exists():
            shutil.copytree(src, target / folder, dirs_exist_ok=True)

    for file_name in INCLUDE_FILES:
        src = source / file_name
        if src.exists():
            shutil.copy2(src, target / file_name)


def remove_private_or_large_artifacts(target: Path) -> None:
    for relative in [
        ".env",
        "data/qdrant",
        "data/neo4j",
        "data/system/classification_log.jsonl",
        "data/system/reflection_quality_log.jsonl",
        "data/system/ingest_51talk_state.json",
        "data/system/classification_rules.json",
        "data/L0/profile.yaml",
        "data/L5/goals.yaml",
        "scripts/seed_51talk_data.sh",
        "scripts/ingest_51talk_worklogs.py",
    ]:
        path = target / relative
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def write_sanitized_templates(target: Path) -> None:
    (target / "data/system").mkdir(parents=True, exist_ok=True)
    (target / "data/L0").mkdir(parents=True, exist_ok=True)
    (target / "data/L5").mkdir(parents=True, exist_ok=True)
    (target / "data/qdrant").mkdir(parents=True, exist_ok=True)
    (target / "data/neo4j").mkdir(parents=True, exist_ok=True)

    (target / ".env.example").write_text(ENV_EXAMPLE_CONTENT, encoding="utf-8")
    (target / "data/L0/profile.yaml").write_text(SANITIZED_PROFILE, encoding="utf-8")
    (target / "data/L5/goals.yaml").write_text(SANITIZED_GOALS, encoding="utf-8")
    (target / "data/system/classification_log.jsonl").write_text("", encoding="utf-8")
    (target / "data/system/reflection_quality_log.jsonl").write_text("", encoding="utf-8")
    (target / "data/system/ingest_51talk_state.json").write_text(
        json.dumps({"last_processed_at": None, "last_source": None}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (target / "data/system/classification_rules.json").write_text(
        json.dumps({"remember": [], "recall": []}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (target / "data/system/classifier_settings.json").write_text(
        json.dumps(
            {
                "learning_enabled": True,
                "vector_enabled": True,
                "llm_fallback_enabled": True,
                "vector_threshold": 0.8,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (target / "data/qdrant/.gitkeep").write_text("", encoding="utf-8")
    (target / "data/neo4j/.gitkeep").write_text("", encoding="utf-8")


def write_public_readme(target: Path) -> None:
    readme = target / "PUBLIC_SANITIZATION.md"
    readme.write_text(
        "\n".join(
            [
                "# Public Sanitization Notes",
                "",
                "This directory is generated from a local private memory-system instance.",
                "It contains code and integration setup, but runtime personal data is removed.",
                "",
                "Sanitized items:",
                "- Removed `.env` and replaced `.env.example` with placeholders",
                "- Replaced `data/L0/profile.yaml` and `data/L5/goals.yaml` with templates",
                "- Cleared system logs under `data/system/*.jsonl`",
                "- Replaced classifier learned rules with empty defaults",
                "- Removed vector/graph runtime storage (`data/qdrant`, `data/neo4j`) and kept `.gitkeep`",
                "",
                "To regenerate: run `python scripts/export_public_sanitized.py --force` in source repo.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def redact_text_files(target: Path) -> None:
    for path in target.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in TEXT_SUFFIX_ALLOWLIST and path.name not in {
            "Dockerfile",
            ".env.example",
            ".gitkeep",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in TEXT_REPLACEMENTS:
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a sanitized public copy.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Source memory-system directory (default: current script parent).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "memory-system-public",
        help="Output directory for sanitized copy.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")
    if source == output:
        raise SystemExit("Source and output must be different directories.")
    if str(output).startswith(str(source) + "/"):
        raise SystemExit("Output must not be inside source directory.")

    if output.exists():
        if not args.force:
            raise SystemExit(f"Output already exists: {output}. Use --force to overwrite.")
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)
    copy_curated_source(source, output)
    remove_private_or_large_artifacts(output)
    write_sanitized_templates(output)
    redact_text_files(output)
    write_public_readme(output)

    print(f"Sanitized copy created: {output}")
    print("Source data remains unchanged.")


if __name__ == "__main__":
    main()
