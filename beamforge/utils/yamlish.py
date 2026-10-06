"""Minimal YAML subset parser (zero-dependency fallback).

Supports the subset we actually emit in configs: nested maps, block lists
(`- item` and `- key: value`), inline scalars, and inline comments (`# ...`).
Used when PyYAML is not installed, so the CLI keeps working on a bare Python.
"""
from collections import OrderedDict


def _strip_comment(line):
    in_q = None
    for i, ch in enumerate(line):
        if ch in ("'", '"'):
            if in_q is None:
                in_q = ch
            elif in_q == ch:
                in_q = None
        elif ch == "#" and in_q is None:
            return line[:i]
    return line


def _parse_scalar(s):
    s = s.strip()
    if s == "":
        return None
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s in ("true", "True", "yes", "on"):
        return True
    if s in ("false", "False", "no", "off"):
        return False
    if s in ("null", "None", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def parse(text):
    lines = text.splitlines()
    root = OrderedDict()
    stack = [(-1, root, None, None)]  # (indent, container, parent, key)

    def _set_val(key, val):
        node = stack[-1][1]
        if isinstance(node, list):
            if isinstance(val, dict):
                node.append(val)
            else:
                node.append(val)
        else:
            node[key] = val

    i = 0
    while i < len(lines):
        raw = lines[i]
        if not raw.strip() or _strip_comment(raw).strip() == "":
            i += 1
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        content = _strip_comment(raw).rstrip()
        stripped = content.strip()
        # block list item
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            # pop to list-level indentation
            while stack and stack[-1][0] >= indent:
                stack.pop()
            container = stack[-1][1]
            if item == "":
                # nested map follows
                child = OrderedDict()
                container.append(child)
                stack.append((indent, child, None, None))
            elif ":" in item:
                k, _, v = item.partition(":")
                child = OrderedDict()
                child[k.strip()] = _parse_scalar(v)
                container.append(child)
            else:
                container.append(_parse_scalar(item))
            i += 1
            continue
        # key: value
        if ":" in content:
            k, _, v = content.partition(":")
            k = k.strip()
            v = v.strip()
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1]
            if v == "":
                child = OrderedDict()
                if isinstance(parent, list):
                    parent.append(child)
                else:
                    parent[k] = child
                stack.append((indent, child, parent, k))
            else:
                if isinstance(parent, list):
                    parent.append(OrderedDict({k: _parse_scalar(v)}))
                else:
                    parent[k] = _parse_scalar(v)
        i += 1
    return dict(root)


def loads(text):
    return parse(text)
