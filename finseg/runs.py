"""실행 기록. **어떤 코드가 만든 자료인지 나중에 반드시 묻게 된다.**"""
import socket
import subprocess
from pathlib import Path

from django.utils import timezone

from finseg.models import Run

REPO = Path(__file__).resolve().parent.parent


def git_sha() -> str:
    """지금 코드의 판. 손댄 것이 있으면 `+dirty` 를 붙인다."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True,
                               timeout=5).stdout.strip()
        return f"{sha}+dirty" if dirty else sha
    except Exception:
        return "?"


def start(kind, model="", params=None, note="") -> Run:
    return Run.objects.create(kind=kind, model=model or "", git_sha=git_sha(),
                              host=socket.gethostname(), params=params or {},
                              note=note or "")


def finish(run: Run) -> None:
    run.finished_at = timezone.now()
    run.save(update_fields=["finished_at"])
