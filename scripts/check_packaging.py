"""Fast manifest checks — versions in lock-step, README declared, licenses present.

The Rust suite proves behavior; this script pins packaging so a version bump
that forgets one manifest fails CI before any tag can (the lesson that
produced OKFgraph's test_packaging.py).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def toml_version(rel: str) -> str | None:
    text = (ROOT / rel).read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else None


def main() -> int:
    failures: list[str] = []

    cargo = toml_version("Cargo.toml")
    pyproject = toml_version("pyproject.toml")
    if not cargo or not pyproject:
        failures.append("version not found in Cargo.toml or pyproject.toml")
    elif cargo != pyproject:
        failures.append(
            f"version lock-step broken: Cargo.toml {cargo} != pyproject.toml {pyproject}"
        )

    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if 'readme = "README.md"' not in pyproject_text:
        failures.append("pyproject.toml does not declare readme = \"README.md\"")

    cargo_text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    for field in ('readme = "README.md"', 'license = "MIT OR Apache-2.0"',
                  'repository = "https://github.com/opticsWolf/embroider"'):
        if field not in cargo_text:
            failures.append(f"Cargo.toml missing {field.split(' =')[0]} metadata")

    readme = ROOT / "README.md"
    if not readme.is_file() or len(readme.read_text(encoding="utf-8")) < 100:
        failures.append("README.md missing or implausibly short")

    for lic in ("LICENSE-APACHE", "LICENSE-MIT"):
        if not (ROOT / lic).is_file():
            failures.append(f"missing {lic}")

    if failures:
        print("packaging check FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"packaging ok: embroider {cargo} (Cargo == pyproject, readme + licenses declared)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
