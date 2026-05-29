"""Load, match, and manage skills.

A skill is a ``SKILL.md`` file: YAML frontmatter (``name``, ``description``, optional
``triggers`` and ``tags``) followed by a Markdown body. We parse the frontmatter with
a deliberately tiny hand-rolled reader so the core install needs no YAML dependency —
it handles the ``key: value`` and list forms our skills use, and degrades gracefully
on anything fancier.

Matching is keyword overlap (the same cheap, transparent approach the memory store
uses for recall): score each skill by how many salient words from the request appear
in its name/description/triggers, and inject the best few. Real semantic matching with
embeddings is an optional upgrade (M10) — this keeps M5 lean, offline, and $0.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_settings

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "and", "for", "what", "with", "this", "that", "are", "was", "you", "your",
    "from", "have", "how", "can", "please", "would", "could", "should", "into", "a",
    "an", "to", "of", "in", "on", "it", "is", "do", "i", "me", "my",
}  # fmt: skip


def _tokens(text: str) -> set[str]:
    """Salient lowercased words from free text (drops stopwords and 1–2 char noise)."""
    return {w for w in _TOKEN.findall(text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


@dataclass(slots=True)
class Skill:
    """One playbook the agent can follow."""

    name: str
    description: str
    body: str = ""
    triggers: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    path: Path | None = None

    def haystack(self) -> str:
        """All the text we match a request against."""
        return " ".join([self.name, self.description, " ".join(self.triggers), " ".join(self.tags)])

    def to_prompt_block(self) -> str:
        """How a matched skill appears in the system prompt."""
        head = f"### {self.name}\n{self.description}".rstrip()
        body = self.body.strip()
        return f"{head}\n{body}" if body else head


# ---- SKILL.md (de)serialization ---------------------------------------------


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Split a ``---`` delimited YAML header from the body. Returns ``(meta, body)``.

    Minimal on purpose: scalars (``key: value``), inline lists (``key: [a, b]``),
    and block lists (``key:`` then ``  - a``). Unknown shapes are kept as strings.
    """
    if not text.lstrip().startswith("---"):
        return {}, text
    # Find the closing fence after the opening one.
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "---")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, text

    meta: dict[str, object] = {}
    current_key: str | None = None
    for raw in lines[start + 1 : end]:
        if not raw.strip():
            continue
        if raw.lstrip().startswith("- ") and current_key:  # block-list item
            meta.setdefault(current_key, [])
            if isinstance(meta[current_key], list):
                meta[current_key].append(_scalar(raw.lstrip()[2:]))
            continue
        if ":" not in raw:
            continue
        key, _, value = raw.partition(":")
        key, value = key.strip(), value.strip()
        current_key = key
        if not value:
            continue  # value arrives as following block-list items
        if value.startswith("[") and value.endswith("]"):
            meta[key] = [_scalar(p) for p in value[1:-1].split(",") if p.strip()]
        else:
            meta[key] = _scalar(value)

    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


def _scalar(value: str) -> str:
    return value.strip().strip("'\"").strip()


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value:
        return [value]
    return []


def parse_skill_md(text: str, path: Path | None = None) -> Skill | None:
    """Parse SKILL.md text into a :class:`Skill` (``None`` if it has no usable name)."""
    meta, body = _parse_frontmatter(text)
    name = str(meta.get("name") or (path.parent.name if path else "")).strip()
    if not name:
        return None
    return Skill(
        name=name,
        description=str(meta.get("description") or "").strip(),
        body=body,
        triggers=_as_list(meta.get("triggers")),
        tags=_as_list(meta.get("tags")),
        path=path,
    )


def render_skill_md(skill: Skill) -> str:
    """Serialize a :class:`Skill` back to SKILL.md text (frontmatter + body)."""
    lines = ["---", f"name: {skill.name}", f"description: {skill.description}"]
    if skill.triggers:
        lines.append(f"triggers: [{', '.join(skill.triggers)}]")
    if skill.tags:
        lines.append(f"tags: [{', '.join(skill.tags)}]")
    lines.append("---")
    return "\n".join(lines) + "\n\n" + skill.body.strip() + "\n"


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "skill"


# ---- the library ------------------------------------------------------------


class SkillLibrary:
    """Holds the skills found under a root directory; matches and persists them.

    Layout: ``<root>/<slug>/SKILL.md`` (the canonical form we write). We also read a
    bare ``<root>/<slug>.md`` so dropping a single file in works too.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self._skills: dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        self._skills.clear()
        if not self.root.exists():
            return
        for md in sorted(self.root.glob("*/SKILL.md")) + sorted(self.root.glob("*.md")):
            try:
                skill = parse_skill_md(md.read_text(encoding="utf-8"), md)
            except OSError:
                continue
            if skill and skill.name not in self._skills:
                self._skills[skill.name] = skill

    # ---- reads ----
    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def count(self) -> int:
        return len(self._skills)

    def match(self, query: str, limit: int = 3) -> list[Skill]:
        """Return the skills most relevant to ``query`` (keyword overlap, best first)."""
        q = _tokens(query)
        if not q:
            return []
        scored: list[tuple[float, Skill]] = []
        for skill in self._skills.values():
            name_tokens = _tokens(skill.name + " " + " ".join(skill.triggers))
            desc_tokens = _tokens(skill.description + " " + " ".join(skill.tags))
            # Name/trigger hits weigh more than description hits.
            score = 2.0 * len(q & name_tokens) + 1.0 * len(q & desc_tokens)
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [skill for _, skill in scored[:limit]]

    # ---- writes ----
    def add(self, skill: Skill) -> Path:
        """Persist a skill as ``<root>/<slug>/SKILL.md`` and register it."""
        folder = self.root / _slug(skill.name)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "SKILL.md"
        path.write_text(render_skill_md(skill), encoding="utf-8")
        skill.path = path
        self._skills[skill.name] = skill
        return path

    def remove(self, name: str) -> bool:
        """Delete a skill by name (removes its folder). Returns True if it existed."""
        skill = self._skills.pop(name, None)
        if skill is None:
            return False
        if skill.path is not None:
            folder = skill.path.parent
            if folder != self.root and folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
            else:
                skill.path.unlink(missing_ok=True)
        return True


# ---- wiring helpers ---------------------------------------------------------


def default_skills_root() -> Path:
    return get_settings().dae_home / "skills"


def _seed_source() -> Path:
    """The starter skills that ship inside the package."""
    return Path(__file__).parent / "library"


def ensure_seeded_skills() -> None:
    """Copy the bundled starter skills into ``~/.dae/skills`` on first run.

    Never overwrites: a skill the user edited or a fresh skill the author wrote
    both survive. Drop the 90-skill Hermes library into this folder and it's picked
    up with zero code changes — the format is identical.
    """
    root = default_skills_root()
    root.mkdir(parents=True, exist_ok=True)
    source = _seed_source()
    if not source.exists():
        return
    for md in source.glob("*.md"):
        dest = root / md.stem / "SKILL.md"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(md.read_text(encoding="utf-8"), encoding="utf-8")


def build_library() -> SkillLibrary:
    """Seed (first run) then load the skill library from ``~/.dae/skills``."""
    ensure_seeded_skills()
    return SkillLibrary(default_skills_root())
