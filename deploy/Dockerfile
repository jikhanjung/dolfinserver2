# dolfinserver2 검토·수집 화면. **자료는 하나도 안 들어간다** — DB·크롭·조각·
# 사진은 전부 호스트 마운트고, 이미지는 코드(1.9MB)와 파이썬 의존성뿐이다.
#
#   docker build -f deploy/Dockerfile -t honestjung/dolfinserver2:X.Y.Z .
#
# 빌드 맥락은 저장소 뿌리다 (`deploy/` 가 아니라). `.dockerignore` 가
# `db/`·`crops/`·`reid/` 를 막는다.
FROM python:3.12-slim

WORKDIR /app

# Pillow(jpeg·webp) · opencv-python-headless(libglib) · gosu(entrypoint 의 권한
# 드롭). headless 판이라 libGL 은 안 든다 — 넣으면 이미지만 커진다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo-dev zlib1g-dev libwebp-dev libglib2.0-0 gosu \
    && gosu nobody true \
    && rm -rf /var/lib/apt/lists/*

# 의존성을 코드보다 먼저 넣는다 — 코드만 바뀐 재빌드가 pip 를 건너뛴다.
COPY requirements.txt deploy/requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-web.txt

COPY . .

RUN chmod +x deploy/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./deploy/entrypoint.sh"]
