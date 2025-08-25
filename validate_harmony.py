#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

def die(msg: str, code: int = 1):
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(code)

def ok(msg: str):
    print(f"✓ {msg}")

def warn(msg: str):
    print(f"⚠ {msg}")

def load_json(p: Path) -> Dict[str, Any]:
    try:
        txt = p.read_text(encoding="utf-8")
    except Exception as e:
        die(f"Failed to read file: {e}")
    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:
        # show a small context window to fix by hand
        context = txt[max(0, e.pos-80):e.pos+80].replace("\n", "\\n")
        die(f"Invalid JSON at line {e.lineno}, col {e.colno}: {e.msg}\n…{context}…")

def validate_harmony(obj: Dict[str, Any]) -> int:
    """Return number of problems found (0 == clean)."""
    problems = 0
    node = obj.get("root", obj)
    if not isinstance(node, dict):
        warn("Top-level is not an object; treating whole document as the finding payload.")
        node = obj

    hrw = node.get("harmony_response_walkthroughs", None)
    if hrw is None:
        warn("Missing 'harmony_response_walkthroughs' key.")
        return problems + 1
    if not isinstance(hrw, list):
        warn("'harmony_response_walkthroughs' is not a list.")
        return problems + 1

    for i, s in enumerate(hrw):
        if not isinstance(s, str):
            warn(f"hrw[{i}] is {type(s).__name__}, expected str.")
            problems += 1
            continue

        # Show how the string will look inside JSON (escaped)
        encoded = json.dumps(s, ensure_ascii=False)

        # Sanity: encoded JSON should NOT contain literal newline characters.
        # (backslash-n sequences are fine)
        if "\n" in encoded:
            warn(f"hrw[{i}] contains literal newline after encoding (should be \\n).")
            problems += 1

        # Optional: quick preview so you can eyeball it
        preview = encoded[:120] + ("…" if len(encoded) > 120 else "")
        print(f"  hrw[{i}] ok | {len(s)} chars | preview: {preview}")

    return problems

def write_canonical(obj: Dict[str, Any], out_path: Path):
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    ok(f"Wrote canonical, properly-escaped JSON → {out_path}")

def main():
    ap = argparse.ArgumentParser(
        description="Validate a single findings JSON and (optionally) write a canonically-escaped copy."
    )
    ap.add_argument("input", help="Path to findings JSON file")
    ap.add_argument("-o", "--out", help="Output path (default: <input>.fixed.json)")
    ap.add_argument("--write", action="store_true", help="Write canonical copy (does not modify input)")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        die(f"File not found: {inp}")

    obj = load_json(inp)
    ok("JSON parsed")

    problems = validate_harmony(obj)
    if problems == 0:
        ok("Harmony strings look properly escaped for Kaggle (\\n, \\\" etc.).")
    else:
        warn(f"Found {problems} issue(s) in harmony strings.")

    if args.write:
        out = Path(args.out) if args.out else inp.with_suffix(inp.suffix + ".fixed.json")
        write_canonical(obj, out)

    # Exit nonzero if there were problems (useful for CI)
    sys.exit(0 if problems == 0 else 2)

if __name__ == "__main__":
    main()
