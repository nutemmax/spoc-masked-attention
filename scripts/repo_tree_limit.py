from pathlib import Path
import random

ROOT = Path(".").resolve()
OVERLEAF = ROOT / "overleaf" / "analysis"

NO_DESCEND = {"results", "logs", "configs", "overleaf"}

IGNORE = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".DS_Store",
    ".venv",
    "venv",
    "wandb",
    "spoc",
    ".fixed-sigma",
    ".old-teacher-attention",
    ".vscode",
}

MAX_PNG_PER_FOLDER = 10
RANDOM_SEED = 42


def should_ignore(path: Path) -> bool:
    return path.name in IGNORE or path.name.startswith(".")


def get_children(path: Path) -> list[Path | str]:
    children = [
        p for p in path.iterdir()
        if not should_ignore(p)
    ]

    pngs = [
        p for p in children
        if p.is_file() and p.suffix.lower() == ".png"
    ]

    non_pngs = [
        p for p in children
        if not (p.is_file() and p.suffix.lower() == ".png")
    ]

    rng = random.Random(RANDOM_SEED + hash(str(path)))
    rng.shuffle(pngs)

    selected_pngs = sorted(
        pngs[:MAX_PNG_PER_FOLDER],
        key=lambda x: x.name.lower(),
    )

    omitted = len(pngs) - len(selected_pngs)

    out: list[Path | str] = sorted(
        non_pngs,
        key=lambda x: (x.is_file(), x.name.lower()),
    )

    out.extend(selected_pngs)

    if omitted > 0:
        out.append(f"[... {omitted} PNG files omitted ...]")

    return out


def tree(path: Path, prefix: str = "", is_root: bool = True) -> None:
    if is_root:
        print(f"{path.name}/")

    children = get_children(path)

    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        connector = "└── " if is_last else "├── "

        if isinstance(child, str):
            print(prefix + connector + child)
            continue

        if child.is_dir() and child.name in NO_DESCEND:
            print(prefix + connector + child.name + "/  [not expanded]")
        else:
            print(prefix + connector + child.name + ("/" if child.is_dir() else ""))
            if child.is_dir():
                extension = "    " if is_last else "│   "
                tree(child, prefix + extension, is_root=False)


if __name__ == "__main__":
    tree(OVERLEAF)