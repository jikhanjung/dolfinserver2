"""**NAS 가 기계마다 다른 자리에 붙는다 — 뿌리는 여기 한 곳이다.**

같은 NAS 인데 보는 길이 다르다. 2080ti 는 WSL 이라 윈도우가 잡은 드라이브를
거쳐 `/mnt/p/JikhanJung/…` 으로 보고, m710q 는 NFS 로 직접 붙여
`/nas/JikhanJung/…` 으로 본다. 그 밑은 양쪽이 같다 —
`dolfinserver2_backup` · `dolfinimage` · `DolFinID/TrainingData`.

**경로를 명령마다 박아 두면 다른 기계에서 "NAS 가 안 붙어 있다" 고 잘못
말한다** — 붙어 있는데 딴 데를 본 것이다. 그 거짓말이 특히 나쁜 자리가
`backup` 이다: 뜰 곳을 못 찾았다는 말과 NAS 가 없다는 말이 같아 보이면
**뜬 줄 알고 지나갈 수 있다.**

새 기계가 늘면 `ROOTS` 에 더한다. `FIN_NAS` 는 언제나 이긴다.
"""
import os
from pathlib import Path

ROOTS = ("/mnt/p/JikhanJung", "/nas/JikhanJung")


def root():
    """붙어 있는 NAS 뿌리. 아무 데도 없으면 첫째를 돌려준다 —
    **없다고 말하는 것은 부르는 쪽의 일이다** (거기가 무엇을 찾는지 안다)."""
    if env := os.environ.get("FIN_NAS"):
        return Path(env)
    for r in ROOTS:
        if Path(r).is_dir():
            return Path(r)
    return Path(ROOTS[0])
