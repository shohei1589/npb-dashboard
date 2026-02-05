# scripts/load_team_pitching.py
import sqlite3
from pathlib import Path
import pandas as pd

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent if THIS.parent.name == "scripts" else THIS.parent
EXCEL_PATH = ROOT / "NPBデータ.xlsx"
DB_PATH = ROOT / "data" / "npb.sqlite"

def load_sheet(sheet_name: str, table_name: str):
    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name).dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]

    required = ["年度", "チーム名", "登板"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{sheet_name} に必要列がありません: {missing} / 実列={list(df.columns)}")

    out = df[["年度", "チーム名", "登板"]].copy()
    out = out.rename(columns={"チーム名": "所属"})
    out["年度"] = pd.to_numeric(out["年度"], errors="coerce").astype("Int64")
    out["登板"] = pd.to_numeric(out["登板"], errors="coerce").fillna(0).astype(int)

    with sqlite3.connect(DB_PATH) as con:
        out.to_sql(table_name, con, if_exists="replace", index=False)

    print(f"✅ {table_name} を作成しました（{sheet_name}）")

def main():
    load_sheet("チーム_投手1", "team_pitching_1")
    load_sheet("チーム_投手2", "team_pitching_2")

if __name__ == "__main__":
    main()
