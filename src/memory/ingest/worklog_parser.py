import re

PROJECT_KEYWORDS = {
    "S9": "51Talk S9培训平台",
    "s9": "51Talk S9培训平台",
    "学情": "51Talk 学情数据",
    "AI API": "51Talk AI API平台",
    "api": "51Talk AI API平台",
    "AIEC": "AIEC 团队项目",
    "Metabase": "学科数据分析",
    "Swagger": "接口验证",
    "GaeaFlow": "发布流水线",
    "流水线": "发布流水线",
}

TEAM_TOKENS = [
    "谢雯萱",
    "刘京楠",
    "AIEC团队",
    "技术开发团队",
    "海外学科",
    "宋超",
    "海涛",
    "吴廖美",
    "王海涛",
    "文静",
    "Rachel舒敏",
]

SKILL_TOKENS = [
    "Metabase",
    "Swagger",
    "SQL",
    "API",
    "Git",
    "GitHub",
    "Cursor",
    "Trae",
    "Neo4j",
    "Qdrant",
    "FastAPI",
]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).replace("```", ""), text)
    text = text.replace("**", "").replace("###", "").replace("##", "").replace("#", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_project_tags(text: str) -> list[str]:
    tags = []
    lower = text.lower()
    for k, tag in PROJECT_KEYWORDS.items():
        if k.lower() in lower and tag not in tags:
            tags.append(tag)
    return tags


def extract_people(text: str) -> list[str]:
    people = []
    for token in TEAM_TOKENS:
        if token in text and token not in people:
            people.append(token)
    return people


def extract_skills(text: str) -> list[str]:
    skills = []
    for token in SKILL_TOKENS:
        if token.lower() in text.lower() and token not in skills:
            skills.append(token)
    return skills


def split_sections(markdown: str) -> list[tuple[str, str]]:
    text = normalize_text(markdown)
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_title = "工作内容"
    current_lines: list[str] = []

    heading_re = re.compile(r"^\s*(?:\d+\.)?\s*([📅🚀💬📚📈⚡📝🏅🌟🎯🔧📊A-Za-z0-9一-龥 \-_/（）()]{2,40})\s*$")

    def flush() -> None:
        nonlocal current_lines
        if current_lines:
            sections.append((current_title, current_lines))
            current_lines = []

    for line in lines:
        striped = line.strip()
        if not striped:
            current_lines.append("")
            continue

        if striped.startswith(("日报", "周报", "工作日报")):
            current_title = "工作概览"
            continue

        if striped.startswith(("开发进度", "沟通与会议", "学习与调研", "调研", "明日计划", "核心工作内容", "数据分析")):
            flush()
            current_title = striped
            continue

        m = heading_re.match(striped)
        if m and len(striped) <= 28 and ("：" not in striped and ":" not in striped):
            flush()
            current_title = striped
            continue

        current_lines.append(striped)

    flush()
    return [(title, "\n".join(chunk).strip()) for title, chunk in sections if "\n".join(chunk).strip()]


def section_to_points(title: str, body: str) -> list[str]:
    points: list[str] = []
    for line in body.split("\n"):
        striped = line.strip()
        if not striped:
            continue
        striped = re.sub(r"^\s*[-*]\s*", "", striped)
        striped = re.sub(r"^\s*\d+[.)]\s*", "", striped)
        if len(striped) < 8:
            continue
        if any(skip in striped for skip in ["报告生成时间", "做得很好", "太棒了"]):
            continue
        points.append(f"{title}：{striped}")

    if not points:
        for p in re.split(r"[。！？]\s*", body):
            p = p.strip()
            if len(p) >= 12:
                points.append(f"{title}：{p}")
    return points


def split_worklog_points(markdown: str, max_points: int = 30) -> list[tuple[str, str]]:
    points: list[tuple[str, str]] = []
    sections = split_sections(markdown)
    for title, body in sections:
        for point in section_to_points(title, body):
            points.append((title, point))
            if len(points) >= max_points:
                return points
    return points


def infer_event_type(title: str, text: str) -> str:
    source = f"{title} {text}"
    if any(k in source for k in ["会议", "沟通", "同步", "讨论"]):
        return "meeting"
    if any(k in source for k in ["调研", "学习", "研究"]):
        return "research"
    if any(k in source for k in ["计划", "明日", "下周"]):
        return "plan"
    return "work"


def infer_layer(title: str, text: str, people: list[str]) -> str:
    src = f"{title} {text}"
    if people or any(k in src for k in ["同事", "团队", "协作", "会议", "沟通"]):
        return "L3"
    if any(k in src for k in ["经验", "方法", "规范", "架构", "标准", "总结", "亮点", "知识"]):
        return "L1"
    return "L2"

