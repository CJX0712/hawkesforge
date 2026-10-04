#!/usr/bin/env python3
"""Phase 0 环境预检（硬门槛）。

用法:
  python preflight.py                 # 人类可读表格 + 退出码 0/1
  python preflight.py --json          # 机器可读 JSON
  python preflight.py --pip-pkg xgboost lightgbm   # 追加轮子先验探测

退出码: 0 = 全通过(或仅 WARN); 1 = 有 BLOCK 级失败(必须先修)。
零依赖，纯 stdlib。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any

# ---- 环境常量（与 SKILL.md §0 保持一致）------------------------------------
HOSTED_PYTHON = r"C:/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe"
VENV_ROOT = r"C:/Users/Administrator/.workbuddy/binaries/python/envs"
GH_OWNER = "CJX0712"
GIT_NAME = "晨星"
GIT_EMAIL = "CJX0712@users.noreply.github.com"
MIN_FREE_GB = 5.0

# 已知无 win_amd64+cp313 预编译 wheel 的黑名单（装不上不许硬编译）
WHEEL_BLACKLIST = {
    "lancedb": "无 cp313 win wheel",
    "llama-cpp-python": "需本地编译 C++/CUDA",
    "torch-geometric": "纯 Python 但依赖 torch；无 torch 时不可用",
    "faiss-gpu": "需 CUDA；本机 CPU-only",
    "scikit-survival": "依赖 ecos 无 cp313 win wheel",
    "karateclub": "GPL-3.0 + 仅 sdist",
    "ray": "Windows 受限环境 raylet/GCS 启动超时",
}


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str
    fix: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, fix: str = "") -> None:
        self.checks.append(Check(name, status, detail, fix))

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "FAIL"]

    @property
    def warned(self) -> list[Check]:
        return [c for c in self.checks if c.status == "WARN"]

    def to_json(self) -> str:
        return json.dumps(
            {
                "ok": not self.failed,
                "checks": [asdict(c) for c in self.checks],
                "summary": {
                    "pass": sum(c.status == "PASS" for c in self.checks),
                    "warn": len(self.warned),
                    "fail": len(self.failed),
                },
            },
            ensure_ascii=False,
            indent=2,
        )


def _run(cmd: list[str], timeout: int = 90) -> tuple[int, str]:
    """跑命令，返回 (rc, combined_output)。不抛异常。"""
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()
    except FileNotFoundError:
        return 127, f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s: {' '.join(cmd)}"
    except Exception as e:  # noqa: BLE001
        return 1, f"{type(e).__name__}: {e}"


# ---- 各项检查 --------------------------------------------------------------
def check_python(rep: Report) -> None:
    py = HOSTED_PYTHON if os.path.exists(HOSTED_PYTHON) else sys.executable
    rc, out = _run([py, "--version"], timeout=30)
    if rc == 0 and out:
        ver = out.strip().split()[-1]
        if not ver.startswith("3.13"):
            rep.add(
                "Python 解释器",
                "WARN",
                f"{py} -> {ver}（非 3.12/3.13，CI 矩阵按 3.12/3.13 收敛）",
                "改用托管 3.13.12",
            )
        else:
            rep.add("Python 解释器", "PASS", f"{py} -> {ver}")
    else:
        rep.add("Python 解释器", "FAIL", f"无法执行 {py}: {out}", "换可用托管版本并在报告中注明")


def check_gh_auth(rep: Report) -> None:
    gh = shutil.which("gh")
    if not gh:
        rep.add("gh 认证", "FAIL", "gh 未安装", "安装 GitHub CLI")
        return
    rc, out = _run([gh, "auth", "status"], timeout=60)
    if rc == 0 and GH_OWNER.lower() in out.lower():
        rep.add("gh 认证", "PASS", f"已登录 {GH_OWNER}")
    elif rc == 0:
        rep.add(
            "gh 认证", "FAIL", f"已登录但不是 {GH_OWNER}: {out[:200]}", "gh auth login 切换账号"
        )
    else:
        rep.add("gh 认证", "FAIL", f"gh auth status 失败: {out[:200]}", "终止，需用户重新登录 gh")


def check_git_identity(rep: Report) -> None:
    rc_n, name = _run(["git", "config", "--global", "user.name"])
    rc_e, email = _run(["git", "config", "--global", "user.email"])
    if rc_n == 0 and name.strip() == GIT_NAME and rc_e == 0 and email.strip() == GIT_EMAIL:
        rep.add("git 署名", "PASS", f"user.name={name.strip()} <{email.strip()}>")
        return
    _run(["git", "config", "--global", "user.name", GIT_NAME])
    _run(["git", "config", "--global", "user.email", GIT_EMAIL])
    rc2, name2 = _run(["git", "config", "--global", "user.name"])
    if rc2 == 0 and name2.strip() == GIT_NAME:
        rep.add("git 署名", "WARN", f"原值 {name!r}/{email!r}，已自动修正为 {GIT_NAME}", "")
    else:
        rep.add("git 署名", "FAIL", "无法写入全局 git 署名", "手动 git config --global")


def check_disk(rep: Report) -> None:
    ws = os.environ.get("WORKBUDDY_CWD") or os.getcwd()
    try:
        usage = shutil.disk_usage(ws)
        free_gb = usage.free / 1024**3
    except Exception as e:  # noqa: BLE001
        rep.add("磁盘空间", "WARN", f"探测失败: {e}")
        return
    if free_gb >= MIN_FREE_GB:
        rep.add("磁盘空间", "PASS", f"{ws} 所在盘剩余 {free_gb:.1f}GB (>= {MIN_FREE_GB}GB)")
    else:
        rep.add(
            "磁盘空间", "FAIL", f"剩余仅 {free_gb:.1f}GB < {MIN_FREE_GB}GB", "清理后重试，否则终止"
        )


def _http_ok(url: str, timeout: int = 20) -> tuple[bool, str]:
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "preflight"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 400, f"HTTP {r.status}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def check_pypi(rep: Report) -> None:
    ok, detail = _http_ok("https://pypi.org/simple/numpy/")
    if ok:
        rep.add("PyPI 可达", "PASS", detail)
    else:
        # 换镜像重试 1 次
        ok2, d2 = _http_ok("https://pypi.tuna.tsinghua.edu.cn/simple/numpy/")
        if ok2:
            rep.add(
                "PyPI 可达",
                "WARN",
                f"pypi.org 不可达({detail})，镜像可达({d2})",
                "后续 pip 走镜像源",
            )
        else:
            rep.add(
                "PyPI 可达",
                "FAIL",
                f"pypi.org={detail}; 镜像={d2}",
                "两次均失败 → 走纯 Tier-1 离线路线",
            )


def check_github(rep: Report) -> None:
    gh = shutil.which("gh")
    if gh:
        rc, out = _run([gh, "api", "user", "--jq", ".login"], timeout=60)
        if rc == 0 and out.strip():
            rep.add("GitHub 可达", "PASS", f"api.github.com OK ({out.strip()})")
        else:
            rep.add(
                "GitHub 可达",
                "WARN",
                f"gh api user 失败: {out[:200]}",
                "推送阶段重试 3 次(间隔 10s)，仍失败则本地交付并标注",
            )
        return
    ok, detail = _http_ok("https://api.github.com")
    rep.add("GitHub 可达", "PASS" if ok else "WARN", detail)


def check_network_split(rep: Report) -> None:
    """本机常见坑：github.com 智能协议被代理 502，但 api.github.com 通。"""
    gh = shutil.which("gh")
    if not gh:
        return
    rc, out = _run([gh, "api", "rate_limit", "--jq", ".rate.remaining"], timeout=45)
    if rc == 0:
        rep.add("GitHub API 通道", "PASS", f"api.github.com 剩余配额 {out.strip()}", "")
        rep.add(
            "git 智能协议",
            "WARN",
            "未验证。若 git push 报 CONNECT 502/reset → 立即改走 gh_push.py 的 Contents API 路线",
            "",
        )
    else:
        rep.add("GitHub API 通道", "WARN", f"rate_limit 查询失败: {out[:160]}")


def check_wheels(rep: Report, packages: list[str]) -> None:
    """轮子先验：py3.13 + win_amd64 是否有可用 wheel。"""
    py = HOSTED_PYTHON if os.path.exists(HOSTED_PYTHON) else sys.executable
    for pkg in packages:
        if pkg.lower() in WHEEL_BLACKLIST:
            rep.add(
                f"轮子先验 {pkg}",
                "WARN",
                f"黑名单：{WHEEL_BLACKLIST[pkg.lower()]} → 不许硬编译，直接换备选",
            )
            continue
        rc, out = _run([py, "-m", "pip", "index", "versions", pkg], timeout=60)
        if rc == 0 and out.strip():
            rep.add(f"轮子先验 {pkg}", "PASS", out.strip().splitlines()[0][:110])
        else:
            rep.add(
                f"轮子先验 {pkg}", "WARN", f"未查到可用版本: {out[:120]}", "换备选或准备离线兜底"
            )


def check_hf_mirror(rep: Report) -> None:
    """HuggingFace 默认不可达；仅在需要预训练权重时才要求镜像。"""
    ok, detail = _http_ok("https://hf-mirror.com", timeout=15)
    rep.add(
        "HF 镜像",
        "WARN" if ok else "PASS",
        f"hf-mirror.com {detail}（仅当选了需在线下载权重的后端才必须可达）",
    )


ICON = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}


def render(rep: Report) -> str:
    lines = ["", "=" * 78, " Phase 0 环境预检", "=" * 78]
    for c in rep.checks:
        lines.append(f"{ICON[c.status]:8} {c.name}")
        lines.append(f"         {c.detail}")
        if c.fix:
            lines.append(f"         处置: {c.fix}")
    p, w, f = sum(c.status == "PASS" for c in rep.checks), len(rep.warned), len(rep.failed)
    lines += ["-" * 78, f" 合计: PASS={p}  WARN={w}  FAIL={f}"]
    lines.append("结论: " + ("可开工" if not f else "存在 BLOCK 级失败，先修再继续"))
    lines += ["=" * 78, ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 0 环境预检")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--pip-pkg", nargs="*", default=[], help="额外做轮子先验探测的包名")
    ap.add_argument("--skip-network", action="store_true", help="跳过网络检查（离线环境）")
    args = ap.parse_args()

    rep = Report()
    check_python(rep)
    check_gh_auth(rep)
    check_git_identity(rep)
    check_disk(rep)
    if not args.skip_network:
        check_pypi(rep)
        check_github(rep)
        check_network_split(rep)
        check_hf_mirror(rep)
    if args.pip_pkg:
        check_wheels(rep, args.pip_pkg)

    if args.json:
        print(rep.to_json())
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        print(render(rep))
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
