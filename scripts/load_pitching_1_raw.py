# scripts/load_pitching_1_raw.py
import sqlite3
from pathlib import Path
import pandas as pd

THIS = Path(__file__).resolve()
# scripts配下でも、project直下でも、常に「projectルート」をROOTにする
ROOT = THIS.parent.parent if THIS.parent.name == "scripts" else THIS.parent
EXCEL_PATH = ROOT / "NPBデータ.xlsx"
DB_PATH = ROOT / "data" / "npb.sqlite"

SHEET_NAME = "一軍基本_投球"  # ←Excelのシート名に合わせて変更

# DB列の順番（必要項目）
COLS = [
    "年度","所属","選手名","選手ID","年齢","投","打",
    "防御率","登板","先発","勝利","敗戦","S","HLD",
    "完投","完封","無四球",
    "被打者","投球回_outs","被安打","被本塁打","四球","敬遠","死球","三振",
    "暴投","ボーク","失点","自責点",
]

def innings_to_outs(x) -> int:
    """
    Excelの投球回表記を outs(int) に変換。
    対応例:
      - 100.1 / 100.2（= 1/3, 2/3）
      - 100.333333 / 100.666667
      - "100 1/3" / "100 2/3"
    """
    if pd.isna(x):
        return 0

    s = str(x).strip()

    # "100 1/3" 形式
    if " " in s and "/" in s:
        ip, frac = s.split(" ", 1)
        ip = int(float(ip))
        frac = frac.strip()
        if frac == "1/3":
            return ip * 3 + 1
        if frac == "2/3":
            return ip * 3 + 2
        return ip * 3

    # 数値として解釈
    try:
        v = float(s)
        ip = int(v)
        frac = v - ip

        # 100.1 / 100.2 形式（= 1/3, 2/3）
        if abs(frac - 0.1) < 1e-6:
            return ip * 3 + 1
        if abs(frac - 0.2) < 1e-6:
            return ip * 3 + 2

        # 100.333333 / 100.666667 形式（= 1/3, 2/3）
        if abs(frac - (1/3)) < 1e-3:
            return ip * 3 + 1
        if abs(frac - (2/3)) < 1e-3:
            return ip * 3 + 2

        # それ以外は整数回扱い
        return ip * 3
    except Exception:
        return 0


def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    id_candidates = ["選手ID", "選手ＩＤ", "player_id", "PlayerID", "ID"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        raise ValueError(f"Excelに選手ID列が見つかりません。候補={id_candidates} / 実列={list(df.columns)}")
    df["選手ID"] = df[id_col].astype(str).str.strip()
    # 余計な空行/空列除去
    df = df.dropna(how="all")

    # 必要なら列名のトリム
    df.columns = [str(c).strip() for c in df.columns]

    # 投球回を outs に変換して保持
    if "投球回" in df.columns and "投球回_outs" not in df.columns:
        df["投球回_outs"] = df["投球回"].apply(innings_to_outs)

    # 数値列を整数に寄せる（空は0）
    int_cols = [
        "登板","先発","勝利","敗戦","S","HLD","完投","完封","無四球",
        "被打者","被安打","被本塁打","四球","敬遠","死球","三振","暴投","ボーク","失点","自責点"
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # 防御率はfloat
    if "防御率" in df.columns:
        df["防御率"] = pd.to_numeric(df["防御率"], errors="coerce")

    # 必須列が足りないときに落とす
    missing = [c for c in ["年度","所属","選手名","選手ID"] if c not in df.columns]
    if missing:
        raise ValueError(f"Excelに必要列がありません: {missing}")

    # DBに入れる列だけに絞る（存在しない列は作る）
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    df = df[COLS].copy()

    with sqlite3.connect(DB_PATH) as con:
        df.to_sql("pitching_1_raw", con, if_exists="replace", index=False)

    print("✅ pitching_1_raw を作成しました")

if __name__ == "__main__":
    main()
