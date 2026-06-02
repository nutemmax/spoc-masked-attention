from pathlib import Path

ROOT = Path(".").resolve()

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


def should_ignore(path: Path) -> bool:
    return path.name in IGNORE or path.name.startswith(".")


def tree(path: Path, prefix: str = "", is_root: bool = True) -> None:
    if is_root:
        print(f"{path.name}/")

    children = [
        p for p in sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        if not should_ignore(p)
    ]

    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        connector = "└── " if is_last else "├── "

        if child.is_dir() and child.name in NO_DESCEND:
            print(prefix + connector + child.name + "/  [not expanded]")
        else:
            print(prefix + connector + child.name + ("/" if child.is_dir() else ""))
            if child.is_dir():
                extension = "    " if is_last else "│   "
                tree(child, prefix + extension, is_root=False)


if __name__ == "__main__":
    tree(ROOT)