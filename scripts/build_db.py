from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd




# ===== 設定 =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = PROJECT_ROOT / "NPBデータ.xlsx"
DB_PATH = PROJECT_ROOT / "data" / "npb.sqlite"

SHEETS = {
    "players": "選手所属",
    "bat1": "一軍基本_打撃",
    "bat2": "二軍基本_打撃",
    "pit1": "一軍基本_投球",
    "pit2": "二軍基本_投球",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名の最低限の正規化（空白除去・全角スペース除去など）"""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\u3000", "", regex=False)  # 全角スペース
        .str.replace(" ", "", regex=False)       # 半角スペース
        .str.strip()
    )
    return df

def innings_to_outs(x) -> int:
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

    # 100.1 / 100.2 形式
    try:
        v = float(s)
        ip = int(v)
        frac = round(v - ip, 3)
        if abs(frac - 0.1) < 1e-6:
            return ip * 3 + 1
        if abs(frac - 0.2) < 1e-6:
            return ip * 3 + 2
        return ip * 3
    except Exception:
        return 0

def read_players_sheet(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    選手所属シート専用の頑丈な読込。
    - header=None で全体を読み込み
    - 「年度」という文字を含む行をヘッダ行として自動検出
    - その行を列名として採用
    """
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    # ヘッダ行候補：行のどこかに「年度」が入っている行
    header_row = None
    for i in range(min(10, len(raw))):
        row = raw.iloc[i].astype(str)
        if row.str.contains("年度", na=False).any():
            header_row = i
            break

    if header_row is None:
        raise ValueError("選手所属シートでヘッダ行（年度を含む行）が見つかりませんでした。")

    header = raw.iloc[header_row].astype(str).tolist()
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = header
    return df


def read_sheet_with_header_row(excel_path: Path, sheet_name: str, header_row: int = 0) -> pd.DataFrame:
    """
    見出し行が通常読込で拾えないExcel向け。
    header=Noneで読み、指定行をcolumnsに採用してデータを切り出す。
    """
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    header = raw.iloc[header_row].astype(str).tolist()
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = header
    return df


def build_players(df: pd.DataFrame) -> pd.DataFrame:
    """年度×選手 のマスタを作る（必要列だけ）"""
    df = normalize_columns(df)

    need = ["年度", "選手名", "選手ID", "所属", "年齢", "投", "打"]

    # --- 追加：列名が取れていない場合、先頭行をヘッダとして採用して復旧する ---
    if not set(need).issubset(set(df.columns)):
        # 先頭行に「年度」「選手名」などが入っている想定（debug_columnsのheadがその状態）
        header = df.iloc[0].astype(str).tolist()

        # headerの正規化（空白・全角スペース除去）
        header = [
            str(x).replace("\u3000", "").replace(" ", "").strip()
            for x in header
        ]

        tmp = df.iloc[1:].copy()
        tmp.columns = header
        tmp = normalize_columns(tmp)

        df = tmp
    # --- 追加ここまで ---

    # 必要列だけ抜く
    df = df[need].copy()

    # 型の軽い整形（失敗しても落ちにくいように）
    df["年度"] = pd.to_numeric(df["年度"], errors="coerce").astype("Int64")
    df["年齢"] = pd.to_numeric(df["年齢"], errors="coerce").astype("Int64")
    df["選手ID"] = df["選手ID"].astype(str)

    # 重複があれば最後を採用（年度×選手IDで一意にしたい）
    df = df.dropna(subset=["年度", "選手ID"])
    df = df.drop_duplicates(subset=["年度", "選手ID"], keep="last")

    return df

def build_batting_1_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    # Excel上で「得点圏」表記の場合はDB側の「得点圏打率」に統一
    if "得点圏" in df.columns and "得点圏打率" not in df.columns:
        df = df.rename(columns={"得点圏": "得点圏打率"})

    need = [
        "年度", "選手名", "選手ID", "所属", "年齢", "投", "打",
        "試合", "打席", "打数", "得点", "安打", "二塁打", "三塁打", "本塁打",
        "塁打", "打点", "三振", "四球", "敬遠", "死球", "犠打", "犠飛",
        "盗塁", "盗塁死", "併殺打", "得点圏打率",
    ]
    df = df[need].copy()

    df["年度"] = pd.to_numeric(df["年度"], errors="coerce").astype("Int64")
    df["選手ID"] = df["選手ID"].astype(str)

    df = df.dropna(subset=["年度", "選手ID"])
    return df



def build_batting_2_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    need = [
        "年度", "選手名", "選手ID", "所属", "年齢", "投", "打",
        "試合", "打席", "打数", "得点", "安打", "二塁打", "三塁打", "本塁打",
        "塁打", "打点", "盗塁", "盗塁死", "犠打", "犠飛", "四球", "敬遠",
        "死球", "三振", "併殺打",
    ]
    df = df[need].copy()
    df["年度"] = pd.to_numeric(df["年度"], errors="coerce").astype("Int64")
    df["選手ID"] = df["選手ID"].astype(str)
    df = df.dropna(subset=["年度", "選手ID"])
    return df


def build_pitching_1_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    # 表記ゆれ寄せ
    rename_map = {}
    if "Ｓ" in df.columns and "S" not in df.columns:
        rename_map["Ｓ"] = "S"
    if "回数" in df.columns and "投球回" not in df.columns:
        rename_map["回数"] = "投球回"
    if "自責" in df.columns and "自責点" not in df.columns:
        rename_map["自責"] = "自責点"
    if "自責点" in df.columns and "自責点" not in df.columns:
        rename_map["自責点"] = "自責点"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "投球回" in df.columns:
        df["投球回_outs"] = df["投球回"].apply(innings_to_outs)

    # ★投球回outsを作る（ソート用）
    if "投球回" in df.columns:
        df["投球回_outs"] = df["投球回"].apply(innings_to_outs)

    # 要件の基本成績（1軍）
    need = [
        "年度", "選手名", "選手ID", "所属", "年齢", "投", "打",
        "防御率", "登板", "先発", "勝利", "敗戦", "S", "HLD",
        "完投", "完封", "無四球",
        "被打者", "投球回", "投球回_outs",
        "被安打", "被本塁打",
        "四球", "敬遠", "死球", "三振",
        "暴投", "ボーク",
        "失点", "自責点",
    ]
    exist = [c for c in need if c in df.columns]
    df = df[exist].copy()

    # 型寄せ
    df["年度"] = pd.to_numeric(df["年度"], errors="coerce").astype("Int64")
    df["選手ID"] = df["選手ID"].astype(str)

    # 整数化したい列
    int_cols = [
        "登板","先発","勝利","敗戦","S","HLD","完投","完封","無四球",
        "被打者","投球回_outs","被安打","被本塁打","四球","敬遠","死球","三振",
        "暴投","ボーク","失点","自責点"
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # 防御率はfloat
    if "防御率" in df.columns:
        df["防御率"] = pd.to_numeric(df["防御率"], errors="coerce")

    df = df.dropna(subset=["年度", "選手ID"])
    return df


def build_pitching_2_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    # 表記ゆれ寄せ（2軍）
    if "自責" in df.columns and "自責点" not in df.columns:
        df = df.rename(columns={"自責": "自責点"})
    if "自責点" in df.columns and "自責点" not in df.columns:
        df = df.rename(columns={"自責点": "自責点"})
    if "打者" in df.columns and "被打者" not in df.columns:
        df = df.rename(columns={"打者": "被打者"})
    if "安打" in df.columns and "被安打" not in df.columns:
        df = df.rename(columns={"安打": "被安打"})
    if "本塁打" in df.columns and "被本塁打" not in df.columns:
        df = df.rename(columns={"本塁打": "被本塁打"})
    if "回数" in df.columns and "投球回" not in df.columns:
        df = df.rename(columns={"回数": "投球回"})

    # ★投球回outs（ソート用）
    if "投球回" in df.columns:
        df["投球回_outs"] = df["投球回"].apply(innings_to_outs)

    # 要件の基本成績（2軍：先発/HLDなし）
    need = [
        "年度", "選手名", "選手ID", "所属", "年齢", "投", "打",
        "防御率", "登板", "勝利", "敗戦", "S",
        "完投", "完封", "無四球",
        "被打者", "投球回", "投球回_outs",
        "被安打", "被本塁打",
        "四球", "敬遠", "死球", "三振",
        "暴投", "ボーク",
        "失点", "自責点",
    ]
    exist = [c for c in need if c in df.columns]
    df = df[exist].copy()

    df["年度"] = pd.to_numeric(df["年度"], errors="coerce").astype("Int64")
    df["選手ID"] = df["選手ID"].astype(str)

    int_cols = [
        "登板","勝利","敗戦","S","完投","完封","無四球",
        "被打者","投球回_outs","被安打","被本塁打","四球","敬遠","死球","三振",
        "暴投","ボーク","失点","自責点"
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "防御率" in df.columns:
        df["防御率"] = pd.to_numeric(df["防御率"], errors="coerce")

    df = df.dropna(subset=["年度", "選手ID"])
    return df


def main() -> None:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excelが見つかりません: {EXCEL_PATH}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    # Excel読み込み
    df_players = read_players_sheet(EXCEL_PATH, SHEETS["players"])
    df_bat1 = pd.read_excel(EXCEL_PATH, sheet_name=SHEETS["bat1"])
    df_bat2 = pd.read_excel(EXCEL_PATH, sheet_name=SHEETS["bat2"])
    df_pit1 = pd.read_excel(EXCEL_PATH, sheet_name=SHEETS["pit1"])
    df_pit2 = pd.read_excel(EXCEL_PATH, sheet_name=SHEETS["pit2"])

    # 整形
    players = build_players(df_players)
    bat1 = build_batting_1_raw(df_bat1)
    bat2 = build_batting_2_raw(df_bat2)
    pit1 = build_pitching_1_raw(df_pit1)
    pit2 = build_pitching_2_raw(df_pit2)

    # SQLiteへ書き込み
    con = sqlite3.connect(DB_PATH)
    try:
        players.to_sql("players", con, if_exists="replace", index=False)
        bat1.to_sql("batting_1_raw", con, if_exists="replace", index=False)
        bat2.to_sql("batting_2_raw", con, if_exists="replace", index=False)
        pit1.to_sql("pitching_1_raw", con, if_exists="replace", index=False)
        pit2.to_sql("pitching_2_raw", con, if_exists="replace", index=False)

        con.execute("CREATE INDEX IF NOT EXISTS idx_players_season_team ON players(年度, 所属)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_bat1_key ON batting_1_raw(年度, 所属)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_bat2_key ON batting_2_raw(年度, 所属)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pit1_key ON pitching_1_raw(年度, 所属)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pit2_key ON pitching_2_raw(年度, 所属)")
        con.commit()
    finally:
        con.close()

    print("✅ DB作成完了:", DB_PATH)
    print("players:", len(players), "bat1:", len(bat1), "bat2:", len(bat2), "pit1:", len(pit1), "pit2:", len(pit2))


if __name__ == "__main__":
    main()
