"""Verify exact public bytes; explicit --write-from-index builds final staged manifests."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import subprocess

EXCLUDED = {"manifests/file_manifest.csv", "manifests/RELEASE_SHA256SUMS.txt", "manifests/PUBLIC_RELEASE_AUDIT_v1.1.1.txt"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--write-from-index", action="store_true", help="Explicit mutation: generate manifests LAST from staged Git blobs.")
    args = parser.parse_args()
    root = args.root.resolve()
    names = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode("utf8").split("\0")
    names = sorted({name for name in names if name} - EXCLUDED)
    if not names:
        raise SystemExit("No tracked files found.")
    manifest = root / "manifests/file_manifest.csv"
    sums = root / "manifests/RELEASE_SHA256SUMS.txt"
    if args.write_from_index:
        rows = []
        for name in names:
            data = subprocess.check_output(["git", "-C", str(root), "show", ":" + name])
            rows.append({"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=["path", "size_bytes", "sha256"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(buffer.getvalue(), encoding="utf8", newline="\n")
        sums.write_text("".join(r["sha256"] + "  " + r["path"] + "\n" for r in rows), encoding="utf8", newline="\n")
        print(f"Wrote {len(rows)} staged-blob records. Stage these files and verify the resulting fresh LF checkout.")
        return
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    sumrows = [line.split("  ", 1) for line in sums.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(rows) != len(names) or len({r["path"] for r in rows}) != len(rows) or {r["path"] for r in rows} != set(names):
        raise SystemExit("file_manifest scope/uniqueness failure.")
    if any(len(r) != 2 for r in sumrows) or len(sumrows) != len(names) or len({r[1] for r in sumrows}) != len(sumrows) or {r[1] for r in sumrows} != set(names):
        raise SystemExit("SHA256SUMS scope/uniqueness failure.")
    hashes = {path: digest for digest, path in sumrows}
    failures = []
    for row in rows:
        path = root / row["path"]
        if not path.is_file():
            failures.append(row["path"] + ": missing")
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != int(row["size_bytes"]) or digest != row["sha256"] or digest != hashes[row["path"]]:
            failures.append(row["path"] + ": exact byte/size mismatch")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {len(names)} unique tracked files, exact bytes and sizes; three documented exclusions.")


if __name__ == "__main__":
    main()
