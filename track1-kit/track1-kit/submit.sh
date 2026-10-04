#!/usr/bin/env bash
# 문제 1 · Autonomous Driver 제출 스크립트 (스타터 킷에 들어 있는 사본)
#
#   ./submit.sh [--dry-run] [--no-check] <제출 ID> <토큰> <제출물 디렉터리>
#
# 제출 ID와 토큰은 참가 안내 메일에 있습니다. 제출물 디렉터리를 킷의 검증기
# (usvnav validate-submission)로 검사하고, zip 하나(track1_<UTC 시각>.zip)로 묶어 올립니다.
# 접수 확인은 끝에 출력되는 "제출 완료: ..." 한 줄입니다. 검증 결과와 점수는 채점이
# 시작된 뒤 사이트에 표시됩니다.
#
#   --dry-run    검증하고 묶기만 하고 올리지 않습니다.
#   --no-check   검증을 건너뜁니다.
#
# 필요한 것: bash, curl, python3 (Windows는 WSL). 검증은 킷을 README대로 설치한 Python에서
# 실행합니다. 기본은 python3이고, 다른 인터프리터는 PYTHON=/경로/python ./submit.sh ...
set -euo pipefail

MAX_MB=5120   # 업로드 크기 상한(MB, 파일 하나 5 GB)
SLOT=track1
EP="https://2kaf47jgwl.execute-api.ap-northeast-2.amazonaws.com/submissions"
PY=${PYTHON:-python3}
KIT=$(CDPATH='' cd "$(dirname "$0")" && pwd)

usage() { echo "사용법: ./submit.sh [--dry-run] [--no-check] <제출 ID> <토큰> <제출물 디렉터리>"; }
die() { echo "오류: $*" >&2; exit 1; }
trim() { local s=$1; s=${s#"${s%%[![:space:]]*}"}; s=${s%"${s##*[![:space:]]}"}; printf '%s' "$s"; }
fsize() { python3 -c 'import os,sys;n=os.path.getsize(sys.argv[1]);print("%.1f MB" % (n/2**20) if n >= 2**20 else "%d KB" % max(1, n//1024))' "$1"; }

DRY=0; CHECK=1; N=0; PID=; TOKEN=; SRC=
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --no-check) CHECK=0 ;;
    -h|--help) usage; exit 0 ;;
    *) N=$((N + 1)); case $N in 1) PID=$a ;; 2) TOKEN=$a ;; 3) SRC=$a ;; esac ;;
  esac
done
[ "$N" -eq 3 ] || { usage >&2; exit 2; }
PID=$(trim "$PID"); TOKEN=$(trim "$TOKEN")   # 메일·CSV에서 복사한 끝의 CR/공백 제거
[ -n "$PID" ] && [ -n "$TOKEN" ] || die "제출 ID와 토큰을 확인하세요."
for t in curl python3; do
  command -v "$t" >/dev/null 2>&1 || die "$t 명령을 찾을 수 없습니다. 설치한 뒤 다시 실행하세요(Windows는 WSL 안에서)."
done
[ -d "$SRC" ] || die "제출물 디렉터리가 아닙니다: $SRC"

# 1) 검증: 킷의 검증기. 통과하지 못하면 올리지 않습니다.
if [ "$CHECK" = 1 ]; then
  PP="$KIT${PYTHONPATH:+:$PYTHONPATH}"
  PYTHONPATH=$PP "$PY" -c 'import usvnav.submission' 2>/dev/null \
    || die "킷의 검증기(usvnav)를 $PY 에서 불러올 수 없습니다. 킷을 README대로 설치하고 그 환경을 활성화한 뒤 다시 실행하세요. 검증 없이 올리려면 --no-check를 붙입니다."
  echo "검증: usvnav validate-submission $SRC"
  PYTHONPATH=$PP "$PY" -m usvnav validate-submission "$SRC" \
    || die "검증을 통과하지 못해 올리지 않았습니다. 위 내용을 고친 뒤 다시 실행하세요."
else
  echo "경고: 검증을 건너뜁니다(--no-check). 형식이 맞지 않는 제출물은 채점에서 거절됩니다." >&2
fi

# 2) 묶기: 임시 디렉터리에 zip 하나(__pycache__/, .git/, .DS_Store 제외)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/submit.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
trap 'exit 1' INT TERM
FILE="$TMP/${SLOT}_$(date -u +%Y%m%dT%H%M%SZ).zip"
python3 - "$SRC" "$FILE" <<'EOF'
import os, sys, zipfile
src, out = sys.argv[1:3]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False) as z:
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".git"))
        for f in sorted(files):
            if f != ".DS_Store":
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, src))
EOF
python3 -c 'import os,sys;sys.exit(os.path.getsize(sys.argv[1]) > int(sys.argv[2]) * 2**20)' "$FILE" "$MAX_MB" \
  || die "제출 파일이 $(fsize "$FILE")로 상한 ${MAX_MB} MB를 넘어 올리지 않았습니다."
echo "제출 파일: $(basename "$FILE") ($(fsize "$FILE"))"
if [ "$DRY" = 1 ]; then echo "--dry-run: 올리지 않았습니다."; exit 0; fi

# 3) 업로드: 운영진 공식 스크립트의 업로드 절차 그대로
NAME=$(basename "$FILE" | tr -c 'A-Za-z0-9._\n-' '_' | sed 's/^[._-]*//' | cut -c1-128)
RESP=$(curl -sf -X POST "$EP" -H "X-Participant-Id: $PID" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d "{\"filename\":\"${NAME:-submission.zip}\"}") \
  || { echo "제출 URL 발급 실패 (제출 ID와 토큰을 확인하세요)" >&2; exit 1; }
ARGS=()
while IFS= read -r line; do ARGS+=(-F "$line"); done < <(python3 -c 'import json,sys;[print(f"{k}={v}") for k,v in json.load(sys.stdin)["fields"].items()]' <<<"$RESP")
URL=$(python3 -c 'import json,sys;print(json.load(sys.stdin)["url"])' <<<"$RESP")
CODE=$(curl -s -o /dev/null -w '%{http_code}' "${ARGS[@]}" -F "file=@$FILE" "$URL")
# shellcheck disable=SC2015  # 공식 스크립트의 줄 그대로
[ "$CODE" = 204 ] && echo "제출 완료: $(python3 -c 'import json,sys;print(json.load(sys.stdin)["key"])' <<<"$RESP")" || { echo "업로드 실패 (HTTP $CODE, ${MAX_MB} MB 초과 여부 확인)" >&2; exit 1; }
echo "검증 결과와 점수는 채점이 시작된 뒤 사이트에 표시됩니다."
