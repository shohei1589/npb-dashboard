import sqlite3
from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "npb.sqlite"
SQL_PATH = PROJECT_ROOT / "scripts" / "create_views.sql"

def main():
    if not SQL_PATH.exists():
        raise FileNotFoundError(f"SQLファイルが見つかりません: {SQL_PATH}")

    sql = SQL_PATH.read_text(encoding="utf-8")

    with sqlite3.connect(DB_PATH) as con:
        con.executescript(sql)

    print("✅ batting_1_view を作成しました")

if __name__ == "__main__":
    main()
