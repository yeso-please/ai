"""개발 RDS에서 ai가 쓰는 백엔드 데이터를 **읽기 전용**으로 내보낸다. 백엔드 데이터의 기준은 개발 RDS다.

내보내는 것 (data/interim/, 커밋 제외)
  descriptions_backend.csv       관광지 설명 (contentid, description)            → build_descriptions.py
  course_overviews_backend.csv   추천코스 소개글 (course_id, overview)             → build_course_index.py

접속
  - RDS 주소는 환경변수 RDS_ENDPOINT로 받는다(레포가 공개라 코드에 넣지 않는다).
  - 계정은 앱 계정 tripin_app, 비밀번호는 backend/config/application-secret.env의 DB_PASSWORD. 출력·명령 인자에 남기지 않는다.
  - 세션을 읽기 전용(default_transaction_read_only=on)으로 강제한다. TLS verify-full, CA는 ~/.aws/rds/global-bundle.pem.
  - 이 PC에 psql이 없어 로컬 Docker 컨테이너(tripin-local-postgres)의 psql을 접속 도구로만 쓴다(로컬 DB는 건드리지 않음).

사용 예:
  RDS_ENDPOINT=tripin-dev-postgres.xxxx.ap-northeast-2.rds.amazonaws.com python scripts/export_from_rds.py
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

CONTAINER = "tripin-local-postgres"
QUERIES = {
    "descriptions_backend.csv": "select source_content_id as contentid, description from app.attractions "
                                "where source_system='TOUR_API' and description is not null and description<>''",
    "course_overviews_backend.csv": "select source_content_id as course_id, description as overview from app.official_courses "
                                    "where source_system='TOUR_API' and description is not null and description<>''",
}


def db_password(env_file: Path) -> str:
    for line in env_file.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*DB_PASSWORD\s*=\s*(.*)", line)
        if m:
            return m.group(1).strip().strip("\"'")
    sys.exit(f"{env_file}에 DB_PASSWORD가 없습니다.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default="../backend/config/application-secret.env")
    parser.add_argument("--ca", default=str(Path.home() / ".aws" / "rds" / "global-bundle.pem"))
    parser.add_argument("--out-dir", default="data/interim")
    args = parser.parse_args()

    endpoint = os.environ.get("RDS_ENDPOINT")
    if not endpoint or not endpoint.endswith(".rds.amazonaws.com") or "prod" in endpoint:
        sys.exit("RDS_ENDPOINT에 개발 RDS 주소(*.rds.amazonaws.com, prod 아님)가 필요합니다.")
    subprocess.run(["docker", "cp", args.ca, f"{CONTAINER}:/tmp/rds-ca.pem"], check=True, capture_output=True)
    env = {**os.environ, "PGPASSWORD": db_password(Path(args.env_file)),
           "PGOPTIONS": "-c default_transaction_read_only=on", "PGCONNECT_TIMEOUT": "10"}
    conn = (f"host={endpoint} port=5432 dbname=tripin_dev user=tripin_app "
            "sslmode=verify-full sslrootcert=/tmp/rds-ca.pem")
    for filename, query in QUERIES.items():
        # 비밀번호는 값 없이 -e PGPASSWORD로 넘긴다(명령 인자에 드러나지 않게).
        r = subprocess.run(["docker", "exec", "-e", "PGPASSWORD", "-e", "PGOPTIONS", "-e", "PGCONNECT_TIMEOUT",
                            CONTAINER, "psql", conn, "-c", f"\\copy ({query}) to stdout with csv header"],
                           env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
        if r.returncode:
            sys.exit(f"{filename} 내보내기 실패: {r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else r.returncode}")
        path = Path(args.out_dir) / filename
        # newline="": Windows에서 설명 안의 \n이 \r\n으로 바뀌지 않게 원문 그대로 쓴다.
        path.write_text(r.stdout, encoding="utf-8", newline="")
        print(f"{path}: {max(r.stdout.count(chr(10)) - 1, 0)}행 (CSV 줄 기준, 설명 안 줄바꿈 포함)")


if __name__ == "__main__":
    main()
