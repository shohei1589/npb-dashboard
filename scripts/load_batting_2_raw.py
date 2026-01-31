import sqlite3
from pathlib import Path
import pandas as pd

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Excel（配置場所が複数あり得るので候補を持つ）
EXCEL_CANDIDATES = [
    PROJECT_ROOT / "NPBデータ.xlsx",
    PROJECT_ROOT / "data" / "NPBデータ.xlsx",
]

DB_PATH = PROJECT_ROOT / "data" / "npb.sqlite"
SHEET_NAME = "二軍基本_打撃"
TABLE_NAME = "batting_2_raw"

def main():
    excel_path = next((p for p in EXCEL_CANDIDATES if p.exists()), None)
    if excel_path is None:
        raise FileNotFoundError(
            "NPBデータ.xlsx が見つかりません。\n"
            f"探した場所: {', '.join(str(p) for p in EXCEL_CANDIDATES)}"
        )

    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite DB が見つかりません: {DB_PATH}")

    # Excel読み込み
    df = pd.read_excel(excel_path, sheet_name=SHEET_NAME)

    # ---- 最低限の列チェック（ここがズレてると後で地獄なので先に止める）----
    required_cols = [
        "年度", "所属", "選手名", "選手ID", "年齢", "投", "打",
        "試合", "打席", "打数", "得点", "安打", "二塁打", "三塁打", "本塁打",
        "塁打", "打点", "三振", "四球", "敬遠", "死球", "犠打", "犠飛",
        "盗塁", "盗塁死", "併殺打",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "二軍基本_打撃 に必要な列が足りません: " + ", ".join(missing)
        )

    # ---- 型を少し整える（SQLiteに入れる前にNaN→0など）----
    # 数値っぽい列は数値化（失敗はNaNになるので最後に埋める）
    numeric_cols = [
        "年度", "年齢",
        "試合", "打席", "打数", "得点", "安打", "二塁打", "三塁打", "本塁打",
        "塁打", "打点", "三振", "四球", "敬遠", "死球", "犠打", "犠飛",
        "盗塁", "盗塁死", "併殺打",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # NaNは0（年齢や年度なども欠損があるなら要確認だが、とりあえず落ちないように）
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # 文字列列の欠損を空文字に
    for c in ["所属", "選手名", "選手ID", "投", "打"]:
        df[c] = df[c].fillna("").astype(str)

    # ---- SQLiteへ書き込み（replaceで作り直す）----
    with sqlite3.connect(DB_PATH) as con:
        df.to_sql(TABLE_NAME, con, if_exists="replace", index=False)

        # 件数確認
        n = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]

    print(f"✅ {TABLE_NAME} を {SHEET_NAME} から作成しました（{n} rows）")
    print(f"   Excel: {excel_path}")
    print(f"   DB:    {DB_PATH}")

if __name__ == "__main__":
    main()
