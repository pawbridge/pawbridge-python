#!/usr/bin/env python3
"""Update the Python AI Service image override in a Helm values file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TAG_PATTERN = re.compile(r"sha-[0-9a-f]{40}")
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
TOP_LEVEL_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+:")
IMAGE_MAPPING_PATTERN = re.compile(r"^image:\s*(?:#.*)?$")
MANAGED_KEY_PATTERN = re.compile(r"^(\s+)(repository|tag|digest):")


def validate_image_reference(repository: str, tag: str, digest: str) -> None:
    if REPOSITORY_PATTERN.fullmatch(repository) is None:
        raise ValueError(f"invalid image repository: {repository}")
    if TAG_PATTERN.fullmatch(tag) is None:
        raise ValueError(f"invalid immutable image tag: {tag}")
    if DIGEST_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"invalid image digest: {digest}")


def update_values(content: str, repository: str, tag: str, digest: str) -> str:
    """Replace the controlled image keys while retaining other YAML content."""
    validate_image_reference(repository, tag, digest)

    newline = "\r\n" if "\r\n" in content else "\n"
    had_final_newline = content.endswith(("\n", "\r"))
    lines = content.splitlines()
    image_lines = [index for index, line in enumerate(lines) if line.startswith("image:")]

    if len(image_lines) > 1:
        raise ValueError("multiple top-level image mappings are not supported")

    managed_lines = [
        f"  repository: {repository}",
        f"  tag: {tag}",
        f"  digest: {digest}",
    ]

    if not image_lines:
        updated_lines = ["image:", *managed_lines, "", *lines]
    else:
        start = image_lines[0]
        if IMAGE_MAPPING_PATTERN.fullmatch(lines[start]) is None:
            raise ValueError("top-level image must be a YAML mapping")

        end = len(lines)
        for index in range(start + 1, len(lines)):
            if TOP_LEVEL_KEY_PATTERN.match(lines[index]):
                end = index
                break

        remaining_block: list[str] = []
        seen_keys: set[str] = set()
        for line in lines[start + 1 : end]:
            match = MANAGED_KEY_PATTERN.match(line)
            if match is None:
                remaining_block.append(line)
                continue

            key = match.group(2)
            if key in seen_keys:
                raise ValueError(f"duplicate image key: {key}")
            seen_keys.add(key)

        updated_lines = [
            *lines[:start],
            lines[start],
            *managed_lines,
            *remaining_block,
            *lines[end:],
        ]

    updated = newline.join(updated_lines)
    if had_final_newline:
        updated += newline
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("values_file", type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--digest", required=True)
    return parser.parse_args()


def update_file(values_file: Path, repository: str, tag: str, digest: str) -> bool:
    """Update one values file without normalizing its existing line endings."""
    with values_file.open("r", encoding="utf-8", newline="") as source:
        original = source.read()

    updated = update_values(original, repository, tag, digest)
    if updated == original:
        return False

    with values_file.open("w", encoding="utf-8", newline="") as destination:
        destination.write(updated)
    return True


def main() -> None:
    args = parse_args()
    changed = update_file(args.values_file, args.repository, args.tag, args.digest)
    if not changed:
        print(f"unchanged: {args.values_file}")
        return

    print(f"updated: {args.values_file}")


if __name__ == "__main__":
    main()
