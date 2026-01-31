import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "npb.sqlite"

con = sqlite3.connect(DB_PATH)

def q(sql: str):
    return pd.read_sql(sql, con)

print("DB:", DB_PATH)

# テーブル一覧
print("\n=== tables ===")
print(q("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"))

# 件数
print("\n=== counts ===")
for t in ["players", "batting_1_raw", "batting_2_raw", "pitching_1_raw", "pitching_2_raw"]:
    print(t, q(f"SELECT COUNT(*) AS n FROM {t}")["n"].iloc[0])

# カラム（先頭20個だけ）
print("\n=== columns batting_1_raw (head 25) ===")
print(q("PRAGMA table_info(batting_1_raw)").head(25)[["name"]].T)

print("\n=== columns pitching_1_raw (head 30) ===")
print(q("PRAGMA table_info(pitching_1_raw)").head(30)[["name"]].T)

# 年度と所属のサンプル
print("\n=== seasons sample ===")
print(q("SELECT 年度, COUNT(*) AS n FROM players GROUP BY 年度 ORDER BY 年度 DESC LIMIT 5"))

print("\n=== teams sample (latest season) ===")
latest = q("SELECT MAX(年度) AS y FROM players")["y"].iloc[0]
print("latest year:", latest)
print(q(f"SELECT 所属, COUNT(*) AS n FROM players WHERE 年度={int(latest)} GROUP BY 所属 ORDER BY n DESC LIMIT 10"))

con.close()
print("\n✅ db_check finished")
