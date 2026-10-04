#!/usr/bin/env python3
"""幂等 GitHub 推送（作者署名 晨星）。

三级降级，自动选可用通道:
  L1  git push --follow-tags        （最快；github.com 智能协议被代理 502 时失败）
  L2  Git Data API                  （blobs->tree->commit->PATCH refs；空仓库先 bootstrap）
  L3  Contents API                  （逐文件 PUT base64；最稳但慢，适合 <200 文件）

用法:
  python gh_push.py --repo myforge --local . --tag v0.1.0 --release "Release notes"
  python gh_push.py --repo myforge --local . --check          # 只探通道，不推
  python gh_push.py --repo myforge --local . --owner CJX0712

零依赖，纯 stdlib + gh CLI。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

OWNER_DEFAULT = "CJX0712"
AUTHOR_NAME = "晨星"
AUTHOR_EMAIL = "CJX0712@users.noreply.github.com"

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "envs",
    ".eggs",
}
SKIP_SUFFIX = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".whl", ".log", ".zip"}
SKIP_NAMES = {".DS_Store", "Thumbs.db", "benchmark.lock"}


# ---- gh 封装（带退避重试，吸收代理 502/bad gateway）-------------------------
class Gh:
    def __init__(self, verbose: bool = True) -> None:
        self.n = 0
        self.verbose = verbose

    def __call__(
        self, *args: str, input_json: dict | None = None, retries: int = 5
    ) -> tuple[int, str, str]:
        cmd = ["gh", *args]
        payload: str | None = None
        tmp: str | None = None
        if input_json is not None:
            # gh api 默认不读 stdin；必须 --input。Windows Git Bash 不支持 <()，落临时文件。
            tmp = os.path.join(
                os.environ.get("TEMP", "."),
                f"_ghpay_{os.getpid()}_{self.n}.json",
            )
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(input_json, f, ensure_ascii=False)
            payload = tmp
            cmd += ["--input", tmp]
        self.n += 1
        last = ""
        for attempt in range(retries):
            try:
                p = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=180,
                )
                out = (p.stdout or "") + (p.stderr or "")
            except subprocess.TimeoutExpired:
                out, p = "timeout", None
                p = None
            finally:
                if tmp and os.path.exists(tmp):
                    for _ in range(5):
                        try:
                            os.unlink(tmp)
                            break
                        except PermissionError:
                            time.sleep(1.0)
            code = p.returncode if p else 124
            last = out
            bad = any(
                k in out.lower()
                for k in (
                    "502",
                    "503",
                    "bad gateway",
                    "reset",
                    "timeout",
                    "econnreset",
                    "unexpected end of json",
                    "tunnel failed",
                )
            )
            if code == 0 or not bad:
                return code, out, " ".join(cmd[:6])
            self.log(f"  retry {attempt + 1}/{retries} (transient): {out.strip()[:120]}")
            time.sleep(2.0 * (attempt + 1))
        return (code if p else 1), last, " ".join(cmd[:6])

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


def collect_files(local: str) -> list[tuple[str, bytes]]:
    root = Path(local).resolve()
    out: list[tuple[str, bytes]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        parts = set(p.relative_to(root).parts)
        if parts & SKIP_DIRS:
            continue
        if p.suffix.lower() in SKIP_SUFFIX or p.name in SKIP_NAMES:
            continue
        if p.stat().st_size > 4 * 1024 * 1024:
            continue
        out.append((p.relative_to(root).as_posix(), p.read_bytes()))
    return out


def probe_channels(gh: Gh, owner: str, repo: str) -> dict[str, bool]:
    res = {}
    rc, out, _ = gh("api", f"repos/{owner}/{repo}")
    res["api"] = rc == 0
    if not res["api"]:
        gh.log(f"  api probe fail: {out[:160]}")
    # git 智能协议探测（只读，不改远端）
    p = subprocess.run(
        ["git", "ls-remote", "--exit-code", "-h", f"https://github.com/{owner}/{repo}.git"],
        capture_output=True,
        text=True,
        timeout=45,
    )
    res["git"] = p.returncode in (0, 2)  # 2 = 空仓库但连通
    if not res["git"]:
        gh.log(f"  git probe fail: {((p.stdout or '') + (p.stderr or ''))[:160]}")
    return res


def ensure_repo(owner: str, repo: str, description: str, gh: Gh) -> None:
    rc, out, _ = gh("repo", "view", f"{owner}/{repo}", "--json", "name")
    if rc == 0 and '"name"' in out:
        gh.log(f"  repo exists: {owner}/{repo}")
        gh("repo", "edit", f"{owner}/{repo}", "--description", description)
        return
    rc, out, _ = gh("repo", "create", f"{owner}/{repo}", "--public", "--description", description)
    if rc != 0:
        raise SystemExit(f"创建仓库失败: {out[:400]}")
    gh.log(f"  repo created: {owner}/{repo}")


def push_git(local: str, owner: str, repo: str, tag: str | None, gh: Gh) -> bool:
    url = f"https://github.com/{owner}/{repo}.git"
    gh("repo", "view", f"{owner}/{repo}")  # noop，保持 gh 上下文
    env = dict(os.environ)
    # 已知：本机 env 代理常导致 CONNECT 502，先试带代理，再试直连
    for unproxy in (False, True):
        e = dict(env)
        if unproxy:
            for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
                e.pop(k, None)
        cmds = [
            ["git", "-C", local, "remote", "remove", "origin"],
            ["git", "-C", local, "remote", "add", "origin", url],
            ["git", "-C", local, "push", "-u", "origin", "HEAD:main"],
        ]
        if tag:
            cmds.append(["git", "-C", local, "tag", "-f", tag])
            cmds.append(["git", "-C", local, "push", "--tags"])
        ok = True
        for c in cmds:
            p = subprocess.run(
                c,
                capture_output=True,
                text=True,
                env=e,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
            if p.returncode != 0 and c[3] != "remove":
                gh.log(
                    f"  git fail ({'直连' if unproxy else '代理'}): "
                    f"{((p.stdout or '') + (p.stderr or ''))[:200]}"
                )
                ok = False
                break
        if ok:
            gh.log("  L1 git push 成功")
            return True
    return False


def _git_data_objects(
    gh: Gh, owner: str, repo: str, files: list[tuple[str, bytes]]
) -> tuple[str | None, str | None]:
    """走 Git Data API：先 bootstrap（空仓库 409），再 blobs->tree->commit->PATCH refs。"""
    base = f"repos/{owner}/{repo}"
    # bootstrap
    rc, out, _ = gh(
        "api",
        f"{base}/contents/.gitkeep",
        "-X",
        "PUT",
        input_json={
            "message": "chore: bootstrap",
            "content": base64.b64encode(b"").decode(),
        },
    )
    head = None
    rc2, out2, _ = gh("api", f"{base}/commits/main")
    if rc2 == 0:
        try:
            head = json.loads(out2).get("sha")
        except json.JSONDecodeError:
            head = None
    parents = [head] if head else []

    tree_entries = []
    for path, data in files:
        rc, out, _ = gh(
            "api",
            f"{base}/git/blobs",
            input_json={"content": base64.b64encode(data).decode(), "encoding": "base64"},
        )
        if rc != 0:
            gh.log(f"  blob fail {path}: {out[:140]}")
            return None, None
        sha = json.loads(out)["sha"]
        tree_entries.append({"path": path, "mode": "100644", "type": "blob", "sha": sha})

    rc, out, _ = gh("api", f"{base}/git/trees", input_json={"tree": tree_entries})
    if rc != 0:
        return None, None
    tree_sha = json.loads(out)["sha"]

    rc, out, _ = gh(
        "api",
        f"{base}/git/commits",
        input_json={
            "message": "feat: full delivery by 晨星",
            "tree": tree_sha,
            "parents": parents,  # 注意：字段是 parents 数组，不是 parent
            "author": {"name": AUTHOR_NAME, "email": AUTHOR_EMAIL},
            "committer": {"name": AUTHOR_NAME, "email": AUTHOR_EMAIL},
        },
    )
    if rc != 0:
        gh.log(f"  commit fail: {out[:200]}")
        return None, None
    commit_sha = json.loads(out)["sha"]

    rc, out, _ = gh(
        "api",
        f"{base}/git/refs/heads/main",
        "-X",
        "PATCH",
        input_json={"sha": commit_sha, "force": True},
    )
    if rc != 0:
        rc, out, _ = gh(
            "api", f"{base}/git/refs", input_json={"ref": "refs/heads/main", "sha": commit_sha}
        )
    if rc != 0:
        return None, None
    return commit_sha, tree_sha


def push_contents_api(
    owner: str, repo: str, local: str, files: list[tuple[str, bytes]], gh: Gh
) -> str | None:
    base = f"repos/{owner}/{repo}/contents"
    last_sha = None
    for path, data in files:
        p = os.path.join(local, path)
        rc, out, _ = gh("api", f"{base}/{path}", "--jq", ".sha")
        cur = out.strip() if rc == 0 else None
        body: dict = {
            "message": f"feat: {path}",
            "content": base64.b64encode(data).decode(),
            "branch": "main",
        }
        if cur:
            body["sha"] = cur
        rc, out, _ = gh("api", f"{base}/{path}", "-X", "PUT", input_json=body)
        if rc != 0:
            gh.log(f"  PUT fail {path}: {out[:140]}")
        if path == "README.md" and rc == 0:
            try:
                last_sha = json.loads(out)["commit"]["sha"]
            except (json.JSONDecodeError, KeyError):
                last_sha = None
        p and None
    rc, out, _ = gh("api", f"repos/{owner}/{repo}/commits/main")
    if rc == 0:
        try:
            return json.loads(out)["sha"]
        except json.JSONDecodeError:
            pass
    return last_sha


def create_tag_release(owner: str, repo: str, tag: str, notes: str, gh: Gh) -> None:
    rc, out, _ = gh("api", f"repos/{owner}/{repo}/git/refs/tags/{tag}", "-X", "DELETE")
    if rc != 0:
        pass
    time.sleep(6)  # ref 删除异步传播，立刻重建会 422
    rc, out, _ = gh(
        "release",
        "create",
        tag,
        "--repo",
        f"{owner}/{repo}",
        "--title",
        tag,
        "--notes",
        notes or tag,
    )
    if rc != 0:
        gh.log(f"  release create fail: {out[:200]}（tag 可能已由 --follow-tags 建好）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--local", default=".")
    ap.add_argument("--owner", default=OWNER_DEFAULT)
    ap.add_argument("--description", default="Delivered by 晨星")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--release", default=None)
    ap.add_argument("--check", action="store_true", help="只探通道")
    ap.add_argument("--force-api", action="store_true", help="跳过 L1，直接走 API")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    gh = Gh()
    print("=" * 70)
    print(f" gh_push  repo={args.owner}/{args.repo}  local={args.local}")
    print("=" * 70)

    ch = probe_channels(gh, args.owner, args.repo)
    print(f" 通道探测: api={ch['api']}  git={ch['git']}")
    if args.check:
        return 0

    ensure_repo(args.owner, args.repo, args.description, gh)
    files = collect_files(args.local)
    print(f" 待推文件: {len(files)} 个")

    pushed = False
    if ch["git"] and not args.force_api:
        pushed = push_git(args.local, args.owner, args.repo, args.tag, gh)

    commit_sha = None
    if not pushed:
        if ch["api"]:
            print(" -> L2 Git Data API")
            commit_sha, _ = _git_data_objects(gh, args.owner, args.repo, files)
            if commit_sha is None:
                print(" -> L3 Contents API (逐文件 PUT)")
                commit_sha = push_contents_api(args.owner, args.repo, args.local, files, gh)
        else:
            print("!! 无可用推送通道 -> 本地交付并在报告中标注")

    if args.tag and commit_sha:
        create_tag_release(args.owner, args.repo, args.tag, args.release or "", gh)

    rc, out, _ = gh("repo", "view", f"{args.owner}/{args.repo}", "--json", "url,defaultBranchRef")
    print("-" * 70)
    print(f" verify: rc={rc} {out.strip()[:200]}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
