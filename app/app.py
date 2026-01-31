import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import math
import streamlit.components.v1 as components

import numpy as np

HILITE_COLS = [
    "打率", "出塁率", "長打率", "OPS",
    "得点圏打率", "wOBA", "BB/K", "Spd",
    "K%", "BB%", "BABIP", "ISO",
]

def pct_rank(series: pd.Series, x: float) -> float:
    """
    series: baseline（該当年度・100打席以上・投手除外済）
    x: 対象セル値
    戻り値: 0.0〜1.0 のパーセンタイル
    """
    if x is None or pd.isna(x):
        return np.nan
    base = pd.to_numeric(series, errors="coerce").dropna().values
    if base.size == 0:
        return np.nan
    base.sort()
    # x以下の要素数 / N
    r = np.searchsorted(base, float(x), side="right") / base.size
    return float(r)

def diverging_color(p: float) -> str:
    """
    0.0(悪い)=青 → 0.5=白 → 1.0(良い)=赤
    Prospectsavantっぽい強めのコントラスト
    """
    if p is None or pd.isna(p):
        return ""
    p = max(0.0, min(1.0, float(p)))

    # 青(#3b82f6) - 白(#ffffff) - 赤(#ef4444)
    blue = np.array([0x3b, 0x82, 0xf6], dtype=float)
    white = np.array([0xff, 0xff, 0xff], dtype=float)
    red  = np.array([0xef, 0x44, 0x44], dtype=float)

    if p < 0.5:
        t = p / 0.5
        rgb = blue*(1-t) + white*t
    else:
        t = (p-0.5) / 0.5
        rgb = white*(1-t) + red*t

    rgb = np.clip(rgb, 0, 255).astype(int)
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


# ===== パス設定 =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "npb.sqlite"

# ===== NPB 球団定義 =====
NPB_TEAMS_1GUN = [
    "ソフトバンク", "日本ハム", "オリックス", "楽天", "西武", "ロッテ",
    "阪神", "巨人", "広島", "DeNA", "中日", "ヤクルト",
    "横浜",  # 2011年以前用
]

TEAM_ABBR = {
    "ソフトバンク": "H",
    "日本ハム": "F",
    "オリックス": "Bs",
    "楽天": "E",
    "西武": "L",
    "ロッテ": "M",
    "阪神": "T",
    "巨人": "G",
    "DeNA": "DB",
    "中日": "D",
    "広島": "C",
    "ヤクルト": "S",
    "横浜": "YB",
    "ハヤテ": "V",
    "オイシックス": "A",
}

NPB_TEAMS_2GUN_EXTRA = ["ハヤテ", "オイシックス"]

# ===== リーグ定義 =====
PACIFIC = ["ソフトバンク", "日本ハム", "オリックス", "楽天", "西武", "ロッテ"]
CENTRAL  = ["阪神", "巨人", "広島", "DeNA", "中日", "ヤクルト"]
# 2011年以前の表記を許容（選択肢としては残す前提）
BAYSTARS_OLD = ["横浜"]

# 2軍（暫定：ここは後で正確な所属に合わせて調整OK）
EASTERN  = ["巨人", "ヤクルト", "DeNA", "楽天", "西武", "日本ハム", "ロッテ", "オイシックス"]
WESTERN  = ["阪神", "広島", "中日", "オリックス", "ソフトバンク", "ハヤテ"]

# ===== 表示カラム定義（まずは1軍打者だけ）=====
BASIC_COLS_BAT1 = [
    "所属",            # すべて/リーグ時のみ値が略称になる（球団指定時は後で落ちる）
    "選手名", "年齢", "投", "打",
    "打率",            # ←「打」と「試合」の間に置く
    "試合", "打席", "打数", "得点", "安打", "二塁打", "三塁打", "本塁打",
    "塁打", "打点", "三振", "四球", "敬遠", "死球", "犠打", "犠飛",
    "盗塁", "盗塁死", "併殺打",
]

ADV_COLS_BAT1 = [
    "所属",
    "選手名", "年齢", "投", "打","打席", 
    "打率", "出塁率", "長打率", "OPS", "得点圏打率",
    "wOBA", "wRC+", "K%", "BB%", "BB/K", "Spd",
    "BABIP", "ISO"
]

# 2軍は得点圏打率が無いので、そこだけ抜いた列セットを用意
BASIC_COLS_BAT2 = BASIC_COLS_BAT1[:]  # 基本は同じ
ADV_COLS_BAT2 = [c for c in ADV_COLS_BAT1 if c != "得点圏打率"]


DISPLAY_COLUMNS = {
    ("1軍", "打者成績", "基本"): BASIC_COLS_BAT1,
    ("1軍", "打者成績", "アドバンスド"): ADV_COLS_BAT1,

    # ★追加（2軍）
    ("2軍", "打者成績", "基本"): BASIC_COLS_BAT2,
    ("2軍", "打者成績", "アドバンスド"): ADV_COLS_BAT2,
}


@st.cache_data
def get_seasons() -> list[int]:
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql("SELECT DISTINCT 年度 FROM players ORDER BY 年度 DESC", con)
    return df["年度"].dropna().astype(int).tolist()


@st.cache_data
def get_teams(season: int, level: str, db_mtime: float) -> list[str]:
    view = "batting_1_view" if level == "1軍" else "batting_2_view"
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql(
            f"SELECT DISTINCT 所属 FROM {view} WHERE 年度 = ? ORDER BY 所属",
            con,
            params=(season,),
        )
    return df["所属"].dropna().astype(str).tolist()



@st.cache_data
def get_batting_1(season: int, team: str) -> pd.DataFrame:
    teams_1gun = tuple(NPB_TEAMS_1GUN)

    if team == "すべて":
        sql = f"""
        SELECT *
        FROM batting_1_view
        WHERE 年度 = ?
          AND 所属 IN {teams_1gun}
        """
        params = (season,)

    elif team == "セリーグ":
        sql = f"""
        SELECT *
        FROM batting_1_view
        WHERE 年度 = ?
          AND 所属 IN {tuple(CENTRAL + BAYSTARS_OLD)}
        """
        params = (season,)

    elif team == "パリーグ":
        sql = f"""
        SELECT *
        FROM batting_1_view
        WHERE 年度 = ?
          AND 所属 IN {tuple(PACIFIC)}
        """
        params = (season,)

    else:
        sql = """
        SELECT *
        FROM batting_1_view
        WHERE 年度 = ? AND 所属 = ?
        """
        params = (season, team)

    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql(sql, con, params=params)

    return df


@st.cache_data
def get_batting_2(season: int, team: str) -> pd.DataFrame:
    teams_2gun = tuple(NPB_TEAMS_1GUN + NPB_TEAMS_2GUN_EXTRA)

    if team == "すべて":
        sql = f"""
        SELECT *
        FROM batting_2_view
        WHERE 年度 = ?
          AND 所属 IN {teams_2gun}
        """
        params = (season,)

    elif team == "イースタン":
        sql = f"""
        SELECT *
        FROM batting_2_view
        WHERE 年度 = ?
          AND 所属 IN {tuple(EASTERN)}
        """
        params = (season,)

    elif team == "ウエスタン":
        sql = f"""
        SELECT *
        FROM batting_2_view
        WHERE 年度 = ?
          AND 所属 IN {tuple(WESTERN)}
        """
        params = (season,)

    else:
        sql = """
        SELECT *
        FROM batting_2_view
        WHERE 年度 = ? AND 所属 = ?
        """
        params = (season, team)

    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql(sql, con, params=params)

    return df


@st.cache_data
def get_pitching_1(season: int, team: str) -> pd.DataFrame:
    sql = """
    SELECT *
    FROM pitching_1_raw
    WHERE 年度 = ? AND 所属 = ?
    """
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql(sql, con, params=(season, team))
    return df


@st.cache_data
def get_pitching_2(season: int, team: str) -> pd.DataFrame:
    sql = """
    SELECT *
    FROM pitching_2_raw
    WHERE 年度 = ? AND 所属 = ?
    """
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql(sql, con, params=(season, team))
    return df

def get_team_choices(season: int, level: str) -> list[str]:
    """
    年度・軍に応じたチーム選択肢を返す
    """
    pacific = ["ソフトバンク", "日本ハム", "オリックス", "楽天", "西武", "ロッテ"]
    central_base = ["阪神", "巨人", "広島", "中日", "ヤクルト"]

    # 横浜 / DeNA 切り替え（表示はそのまま）
    if season <= 2011:
        central = central_base + ["横浜"]
    else:
        central = central_base + ["DeNA"]

    teams = pacific + central

    # 2軍：2024年以降に追加
    if level == "2軍" and season >= 2024:
        teams = teams + ["ハヤテ", "オイシックス"]
    
    teams = ["すべて"] + teams
    return teams

def normalize_col(x: object) -> str:
    s = str(x)
    # 不可視系を除去
    s = s.replace("\ufeff", "")   # BOM
    s = s.replace("\u200b", "")   # zero-width space
    s = s.replace("\xa0", " ")    # NBSP
    # 全角→半角
    s = s.replace("％", "%")
    # 前後空白
    s = s.strip()
    return s


st.set_page_config(page_title="年度成績ダッシュボード", layout="wide")
st.title("年度成績ダッシュボード")

# ===== 見た目（ヘッダー用CSS）=====
st.markdown(
    """
    <style>
      .hero {
        padding: 18px 18px 14px 18px;
        border: 1px solid rgba(49,51,63,0.12);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(16,185,129,0.08));
        margin-bottom: 12px;
      }
      .hero-title {
        font-size: 34px;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.02em;
      }
      .hero-sub {
        margin-top: 6px;
        font-size: 14px;
        opacity: 0.75;
      }
      /* dataframe内のセルを中央寄せ */
      .stDataFrame td, .stDataFrame th { text-align: center !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===== DataFrame表示用CSS（確実に中央寄せ）=====
st.markdown(
    """
    <style>
    /* StreamlitのDataFrameコンポーネント配下に限定して強制上書き */
    [data-testid="stDataFrame"] * {
        font-family: Meiryo, "メイリオ", "Hiragino Kaku Gothic ProN", "Noto Sans JP", sans-serif !important;
    }

    /* ヘッダ */
    [data-testid="stDataFrame"] thead th {
        text-align: center !important;
        vertical-align: middle !important;
        white-space: nowrap !important;
    }

    /* セル */
    [data-testid="stDataFrame"] tbody td {
        text-align: center !important;
        vertical-align: middle !important;
        white-space: nowrap !important;
    }

    /* 1列目（選手名）だけ左寄せ */
    [data-testid="stDataFrame"] tbody td:first-child {
        text-align: left !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ===== 全体CSS（フォント・中央寄せなど）=====
st.markdown(
    """
    <style>
      html, body, [class*="css"]  {
        font-family: Meiryo, "メイリオ", "Hiragino Kaku Gothic ProN", "Noto Sans JP", sans-serif;
      }
      /* dataframe内のセルを中央寄せ */
      .stDataFrame td, .stDataFrame th {
        text-align: center !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===== 上部ナビ（1軍/2軍、打者/投手）=====
# Streamlitのバージョンによって segmented_control が無い場合があるので保険付き
try:
    col1, col2 = st.columns([1, 2])

    with col1:
        level = st.segmented_control(
            "軍",
            options=["1軍", "2軍"],
            default="1軍",
            label_visibility="collapsed",
        )

    with col2:
        category = st.segmented_control(
            "カテゴリ",
            options=["打者成績", "投手成績"],  # まずは2つ
            default="打者成績",
            label_visibility="collapsed",
        )
except Exception:
    # フォールバック（segmented_controlが無い場合）
    tabs = st.tabs(["1軍", "2軍"])
    with tabs[0]:
        level = "1軍"
    with tabs[1]:
        level = "2軍"

    category = st.radio("カテゴリ", ["打者成績", "投手成績"], horizontal=True, label_visibility="collapsed")

# ===== 上部フィルタ（年度・所属）=====
seasons = get_seasons()
if not seasons:
    st.error("年度が取得できませんでした。DB内容を確認してください。")
    st.stop()

# 上部で年度・所属を並べる
colA, colB, colC = st.columns([1, 1, 4])

# session_state の初期値（デフォルト：2025年 / ソフトバンク）
DEFAULT_SEASON = 2025
DEFAULT_TEAM = "ソフトバンク"

if "season" not in st.session_state:
    st.session_state["season"] = DEFAULT_SEASON if DEFAULT_SEASON in seasons else seasons[0]

if "team" not in st.session_state:
    st.session_state["team"] = DEFAULT_TEAM

with colA:
    st.selectbox(
        "年度",
        seasons,
        key="season",   # session_state["season"] を値として使う
    )

# ===== 所属選択肢（すべて/リーグ/球団）=====
if level == "1軍":
    teams = ["すべて", "セリーグ", "パリーグ"] + CENTRAL + PACIFIC + BAYSTARS_OLD
else:
    teams = ["すべて", "イースタン", "ウエスタン"] + EASTERN + WESTERN

# ===== 横浜 ↔ DeNA 自動リダイレクト（あなたの要件：逆もしかり）=====
season_now = st.session_state["season"]
team_now = st.session_state.get("team")

if team_now in ("DeNA", "横浜"):
    if season_now <= 2011 and team_now == "DeNA":
        st.session_state["team"] = "横浜"
    elif season_now >= 2012 and team_now == "横浜":
        st.session_state["team"] = "DeNA"

# ===== 存在しないチームだったらフォールバック =====
if "team" not in st.session_state:
    st.session_state["team"] = "ソフトバンク"

if st.session_state["team"] not in teams and teams:
    st.session_state["team"] = teams[0]

with colB:
    st.selectbox("所属", teams, key="team")

season = st.session_state["season"]
team = st.session_state["team"]


with colC:
    show_mode = st.radio(
        "表示",
        ["基本", "アドバンスド"],
        horizontal=True,
        index=0,
        key="show_mode",
        label_visibility="collapsed",
    )

    hide_pitchers = st.checkbox(
        "投手を除外",
        value=True,
        key="hide_pitchers",
    )

    # ===== 打席フィルタのデフォルト制御（team変更時に追従）=====
    broad_teams = ("すべて", "セリーグ", "パリーグ", "イースタン", "ウエスタン")
    is_broad_team = st.session_state.get("team") in broad_teams

    # team が変わった瞬間を検知
    prev_team = st.session_state.get("_prev_team")
    now_team = st.session_state.get("team")
    team_changed = (prev_team is not None) and (prev_team != now_team)
    st.session_state["_prev_team"] = now_team

    # teamが「広い選択」に変わったら規定打席へ寄せる
    if "pa_filter" not in st.session_state:
        st.session_state["pa_filter"] = "規定打席" if is_broad_team else "すべて"
    else:
        if team_changed and is_broad_team:
            st.session_state["pa_filter"] = "規定打席"

    # ===== 打席フィルタ UI =====
    pa_options = ["規定打席", "400", "300", "200", "100", "50", "すべて"]
    st.selectbox(
        "打席フィルタ",
        pa_options,
        key="pa_filter",
    )

# ===== 表示（上部ナビに応じて切り替え）=====
st.markdown(f"### {level}・{category}")

if level == "1軍" and category == "打者成績":
    df = get_batting_1(season, team)   # batting_1_view
elif level == "2軍" and category == "打者成績":
    df = get_batting_2(season, team)   # batting_2_view
elif level == "1軍" and category == "投手成績":
    df = get_pitching_1(season, team)
elif level == "2軍" and category == "投手成績":
    df = get_pitching_2(season, team)
else:
    df = pd.DataFrame()


# ===== ハイライト用の母集団（該当年度の全選手、100打席以上） =====
# ※要件：チームで絞って表示していても、比較は年度全体
if level == "1軍":
    baseline = get_batting_1(season, "すべて").copy()
    pitching_all_table = "pitching_1_raw"
else:
    baseline = get_batting_2(season, "すべて").copy()
    pitching_all_table = "pitching_2_raw"

# 投手除外がONなら、母集団も同じ条件で投手除外（年度全体で判定）
# ===== ハイライト用の母集団（該当年度の全選手、100打席以上） =====
# ※要件：チームで絞って表示していても、比較は年度全体
# ※2軍にも対応：levelで参照先を切り替える

if category == "打者成績":
    # 年度全体の打者母集団を作る（表示チームに関係なく比較する）
    if level == "1軍":
        baseline = get_batting_1(season, "すべて").copy()
        pitching_table_all = "pitching_1_raw"
    else:
        baseline = get_batting_2(season, "すべて").copy()
        pitching_table_all = "pitching_2_raw"
else:
    baseline = pd.DataFrame()

# 投手除外がONなら、母集団も同じ条件で投手除外（年度全体で判定）
if category == "打者成績" and st.session_state.get("hide_pitchers", True) and not baseline.empty:
    with sqlite3.connect(DB_PATH) as con:
        df_p_all = pd.read_sql(
            f"""
            SELECT 選手ID, SUM(COALESCE(登板,0)) AS 登板
            FROM {pitching_table_all}
            WHERE 年度 = ?
            GROUP BY 選手ID
            """,
            con,
            params=(season,),
        )

    df_p_all["登板"] = pd.to_numeric(df_p_all["登板"], errors="coerce").fillna(0)

    # baseline側の試合も数値化
    baseline["試合"] = pd.to_numeric(baseline.get("試合", 0), errors="coerce").fillna(0)

    # 選手IDで結合して投手判定
    if "選手ID" in baseline.columns:
        baseline = baseline.merge(df_p_all, on="選手ID", how="left")
        baseline["登板"] = baseline["登板"].fillna(0)

        # 判定：登板×1.2 > 試合 → 投手扱い
        is_pitcher = (baseline["登板"] * 1.2) > baseline["試合"]
        baseline = baseline.loc[~is_pitcher].copy()
        baseline = baseline.drop(columns=["登板"], errors="ignore")

# 100打席以上で母集団固定（ここは要件通り）
if category == "打者成績" and not baseline.empty:
    baseline["打席"] = pd.to_numeric(baseline.get("打席", 0), errors="coerce").fillna(0)
    baseline = baseline[baseline["打席"] >= 100].copy()



# ===== ここから 1軍打者の整形（崩れない順番で固定）=====
mode = st.session_state.get("show_mode", "基本")
show_team_col = team in ("すべて", "セリーグ", "パリーグ", "イースタン", "ウエスタン")

# ---- 1) 所属（略称）表示：列名は「所属」、中身だけ略称にする ----
if (category == "打者成績") and show_team_col and ("所属" in df.columns):
    df["所属"] = df["所属"].astype(str).str.strip()
    df["所属"] = df["所属"].map(TEAM_ABBR).fillna(df["所属"])

# ---- 2) 年度は常に落とす。所属は球団指定時のみ落とす ----
drop_cols = []
if "年度" in df.columns:
    drop_cols.append("年度")
if (not show_team_col) and ("所属" in df.columns):
    drop_cols.append("所属")
if drop_cols:
    df = df.drop(columns=drop_cols, errors="ignore")

# ---- 3) 投手除外（打者成績のみ）※選手IDが必要なので、ここでは落とさない ----
if category == "打者成績" and st.session_state.get("hide_pitchers", True) and not df.empty:
    pitching_table = "pitching_1_raw" if level == "1軍" else "pitching_2_raw"

    # 「すべて/リーグ」の時は所属が複数なので、ここは安全のためスキップ（重い & ロジック複雑）
    # まずは球団指定時のみ投手除外を効かせる
    if (not show_team_col) and ("選手ID" in df.columns):
        sql_p = f"""
        SELECT 選手ID, 登板
        FROM {pitching_table}
        WHERE 年度 = ? AND 所属 = ?
        """
        with sqlite3.connect(DB_PATH) as con:
            df_p = pd.read_sql(sql_p, con, params=(season, team))

        df_p["登板"] = pd.to_numeric(df_p["登板"], errors="coerce").fillna(0)

        if "試合" in df.columns:
            df["試合"] = pd.to_numeric(df["試合"], errors="coerce").fillna(0)
        else:
            df["試合"] = 0

        df = df.merge(df_p, on="選手ID", how="left")
        df["登板"] = df["登板"].fillna(0)

        is_pitcher = (df["登板"] * 1.2) > df["試合"]
        df = df.loc[~is_pitcher].copy()

        df = df.drop(columns=["登板"], errors="ignore")

# ---- 4) 打席フィルタ（数値化してから） ----
pa_filter = st.session_state.get("pa_filter", "すべて")

if "打席" in df.columns:
    df["打席"] = pd.to_numeric(df["打席"], errors="coerce").fillna(0)

if "打席" in df.columns and pa_filter != "すべて":
    if pa_filter in ["50", "100", "200", "300", "400"]:
        thr = float(pa_filter)
        df = df[df["打席"] >= thr]
    elif pa_filter == "規定打席":
        factor = 3.1 if level == "1軍" else 2.7
        standard_games = 143 if level == "1軍" else 120
        threshold = math.floor(standard_games * factor)
        df = df[df["打席"] >= threshold]

# ---- 5) デフォルトソート（打者：打席降順） ----
if level in ("1軍", "2軍") and category == "打者成績" and "打席" in df.columns:
    df = df.sort_values("打席", ascending=False, na_position="last")

# ---- 6) 列出し分け（基本/アドバンスド） ----
if level in ("1軍", "2軍") and category == "打者成績":
    key = (level, category, mode)
    if key in DISPLAY_COLUMNS:
        cols = DISPLAY_COLUMNS[key]

        # 得点圏打率は 1軍のみ 2016年未満で落とす（2軍はそもそも列セットに無い）
        if level == "1軍" and season < 2016 and "得点圏打率" in cols:
            cols = [c for c in cols if c != "得点圏打率"]

        cols_exist = [c for c in cols if c in df.columns]
        df = df[cols_exist]

# ---- 7) ここで「選手ID」を確実に非表示（でも上の処理では使える） ----
df = df.drop(columns=["選手ID"], errors="ignore")

# ===== 表示フォーマット（Stylerでまとめて）=====

# 0) 列名正規化（df / baseline 両方）
df.columns = [normalize_col(c) for c in df.columns]
baseline.columns = [normalize_col(c) for c in baseline.columns]

RATE_COLUMNS = ["打率", "出塁率", "長打率", "OPS", "wOBA", "BABIP", "ISO", "得点圏打率"]
PCT_COLUMNS  = ["K%", "BB%"]

def fmt_rate_dot(x):
    if x is None or pd.isna(x):
        return "-"
    try:
        s = f"{float(x):.3f}"
    except Exception:
        return "-"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s

def fmt_percent_1(x):
    if x is None or pd.isna(x):
        return "-"
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "-"

# 1) 整数列
INT_COLS = [
    "年齢", "試合", "打席", "打数", "得点", "安打", "二塁打", "三塁打", "本塁打",
    "塁打", "打点", "三振", "四球", "敬遠", "死球", "犠打", "犠飛",
    "盗塁", "盗塁死", "併殺打",
]
int_cols_exist = [c for c in INT_COLS if c in df.columns]
for c in int_cols_exist:
    df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")

# 2) ヒートマップ対象列を数値化（df / baseline 両方）
for c in HILITE_COLS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if c in baseline.columns:
        baseline[c] = pd.to_numeric(baseline[c], errors="coerce")

# K% は低い方が良いので反転（必要ならここに追加）
reverse_cols = {"K%"}  # 例：防御率系なども「低いほど良い」なら追加

def apply_heatmap(data: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame("", index=data.index, columns=data.columns)

    # ✅ 100打席以上だけ色付け（表示df側）
    if "打席" in data.columns:
        mask_100pa = pd.to_numeric(data["打席"], errors="coerce").fillna(0) >= 100
    else:
        mask_100pa = pd.Series(False, index=data.index)

    # ✅ 「基本」では打率だけ色を付けない（あなたの要望）
    mode = st.session_state.get("show_mode", "基本")
    hilite_cols = HILITE_COLS.copy()
    if mode == "基本" and "打率" in hilite_cols:
        hilite_cols.remove("打率")

    for col in hilite_cols:
        if col not in data.columns:
            continue
        if col not in baseline.columns:
            continue

        base_series = pd.to_numeric(baseline[col], errors="coerce")

        # 1セルずつパーセンタイル→色
        ps = data[col].apply(lambda x: pct_rank(base_series, x))

        # 反転（小さいほど良い指標）
        if col in reverse_cols:
            ps = 1.0 - ps

        colors = ps.apply(diverging_color)

        # ✅ 100打席未満は無色にする
        colors = colors.where(mask_100pa, "")

        out[col] = colors.apply(
            lambda c: "" if c == "" else f"background-color: {c}; font-weight: 600;"
        )

    return out


# 3) fmt辞書（表示形式）
fmt = {}

if "BB/K" in df.columns:
    df["BB/K"] = pd.to_numeric(df["BB/K"], errors="coerce")
    fmt["BB/K"] = lambda x: "-" if pd.isna(x) else f"{float(x):.2f}"

if "Spd" in df.columns:
    df["Spd"] = pd.to_numeric(df["Spd"], errors="coerce")
    fmt["Spd"] = lambda x: "-" if pd.isna(x) else f"{float(x):.1f}"


# 整数
for c in int_cols_exist:
    fmt[c] = (lambda v: "" if pd.isna(v) else f"{int(v)}")

# 率（.xxx）
for c in RATE_COLUMNS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        fmt[c] = fmt_rate_dot

# %（xx.x%）
for c in PCT_COLUMNS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        fmt[c] = fmt_percent_1

# 4) table_styles（ここで定義してから使う）
table_styles = [
    {"selector": "th", "props": [("text-align", "center !important")]},
    {"selector": "td", "props": [("text-align", "center !important")]},
]
# 「所属」列があるなら 2列目が選手名、無いなら 1列目が選手名
if "所属" in df.columns:
    table_styles.append({"selector": "tbody tr td:nth-child(2)", "props": [("text-align", "left !important")]})
else:
    table_styles.append({"selector": "tbody tr td:nth-child(1)", "props": [("text-align", "left !important")]})

# 5) Styler：順番を固定（apply → format → styles）
styler = df.style
styler = styler.apply(apply_heatmap, axis=None)  # 背景色（数値が必要）
styler = styler.format(fmt)                      # 表示形式
styler = styler.set_table_styles(table_styles)   # CSS

html_table = styler.hide(axis="index").to_html()


full_html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root {{
    --border: #d1d5db;
    --border2: #e5e7eb;
    --text: #111827;
    --shadow: 0 8px 24px rgba(17,24,39,0.08);
    --radius: 14px;

    /* PC基準 */
    --th-font: 13px;
    --td-font: 14px;
    --th-pad-y: 8px;
    --th-pad-x: 10px;
    --td-pad-y: 7px;
    --td-pad-x: 10px;

    /* 打率以降の統一列幅（PC） */
    --w-metric: 78px;

    /* ★選手名列の幅（PC）＝「5文字が入る」くらい */
    --w-name: 96px;

    /* ★所属列の幅（PC）＝「2文字が入る」くらい */
    --w-team: 44px;
  }}

  body {{
    margin: 0;
    background: #ffffff;
    color: var(--text);
    font-family: Meiryo, "メイリオ", "Hiragino Kaku Gothic ProN", "Noto Sans JP", sans-serif;
  }}

  .npb-table-wrap {{
    overflow: auto;
    max-height: 86vh;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: #ffffff;
    box-shadow: var(--shadow);
  }}

  table {{
    border-collapse: collapse;
    width: 100%;
  }}

  thead th {{
    position: sticky;
    top: 0;
    z-index: 3;
    background: #ffffff;
    border: 1px solid var(--border);
    padding: var(--th-pad-y) var(--th-pad-x);
    font-size: var(--th-font);
    font-weight: 600;
    white-space: nowrap;
    text-align: center;
    cursor: pointer;
  }}

  tbody td {{
    border: 1px solid var(--border2);
    padding: var(--td-pad-y) var(--td-pad-x);
    font-size: var(--td-font);
    white-space: nowrap;
    text-align: center;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  tbody tr:hover td {{
    background: rgba(37,99,235,0.06);
  }}

  /* ===== スマホだけ：文字/余白/列幅を小さく ===== */
  @media (max-width: 768px) {{
    :root {{
      --th-font: 11px;
      --td-font: 12px;
      --th-pad-y: 6px;
      --th-pad-x: 7px;
      --td-pad-y: 5px;
      --td-pad-x: 7px;

      /* スマホは少し細く（横スクロール前提） */
      --w-metric: 66px;

      /* ★選手名列（スマホ） */
      --w-name: 84px;

      /* ★所属列（スマホ） */
      --w-team: 38px;
    }}
  }}
</style>
</head>

<body>
  <div class="npb-table-wrap">
    {html_table}
  </div>

<script>
(function() {{
  function getValue(row, idx) {{
    const t = row.children[idx].innerText.trim();
    if (t === "-" || t === "") return null;
    const n = Number(t.replace("%","").replace(/^\\./,"0."));
    return isNaN(n) ? t : n;
  }}

  function sortTable(table, col, asc) {{
    const tb = table.tBodies[0];
    const rows = Array.from(tb.rows);
    rows.sort((a, b) => {{
      const A = getValue(a, col);
      const B = getValue(b, col);
      if (A === null && B === null) return 0;
      if (A === null) return 1;
      if (B === null) return -1;
      if (typeof A === "number" && typeof B === "number") {{
        return asc ? A - B : B - A;
      }}
      return asc
        ? String(A).localeCompare(String(B), "ja")
        : String(B).localeCompare(String(A), "ja");
    }});
    rows.forEach(r => tb.appendChild(r));
  }}

  function injectColWidthStyle(colIndex1Based, px, extra="") {{
    const css = document.createElement("style");
    css.textContent = `
      thead th:nth-child(${{colIndex1Based}}),
      tbody td:nth-child(${{colIndex1Based}}) {{
        width: ${{px}}px !important;
        min-width: ${{px}}px !important;
        max-width: ${{px}}px !important;
        ${{extra}}
      }}
    `;
    document.head.appendChild(css);
  }}

  function freezeColumns(ths, headerToIndex, names) {{
    const existing = names.filter(n => headerToIndex.has(n));
    if (existing.length === 0) return;

    let left = 0;
    existing.forEach((name, k) => {{
      const idx1 = headerToIndex.get(name);   // 1-based
      const idx0 = idx1 - 1;                  // 0-based
      const th = ths[idx0];
      if (!th) return;

      const w = Math.ceil(th.getBoundingClientRect().width);
      const z = 60 - k;
      const shadowCss = (k === existing.length - 1)
        ? "box-shadow: 6px 0 8px rgba(17,24,39,0.10);"
        : "";

      const css = document.createElement("style");
      css.textContent = `
        thead th:nth-child(${{idx1}}),
        tbody td:nth-child(${{idx1}}) {{
          position: sticky !important;
          left: ${{left}}px !important;
          z-index: ${{z}} !important;
          background: rgba(255,255,255,0.98) !important;
          ${{shadowCss}}
        }}
      `;
      document.head.appendChild(css);

      left += w;
    }});
  }}

  function autoShrinkNameCells(table, nameIdx1) {{
    // 「5文字が入る幅」を優先し、溢れるときだけ文字を縮小
    const base = parseInt(getComputedStyle(document.documentElement).getPropertyValue("--td-font")) || 14;
    const minSize = 9;  // これ以下にはしない
    const cells = Array.from(table.querySelectorAll(`tbody td:nth-child(${{nameIdx1}})`));

    cells.forEach(td => {{
      const text = td.innerText.trim();
      if (!text) return;

      const len = text.length;

      // 5文字以内ならベースのまま
      if (len <= 5) {{
        td.style.fontSize = `${{base}}px`;
        return;
      }}

      // 6文字以上は段階的に縮める（自然な範囲で）
      const newSize = Math.max(minSize, base - (len - 5));
      td.style.fontSize = `${{newSize}}px`;
    }});
  }}

  function bind() {{
    const table = document.querySelector("table");
    if (!table) return;

    const ths = Array.from(table.querySelectorAll("thead th"));
    const headerToIndex = new Map();
    ths.forEach((th, i) => headerToIndex.set(th.innerText.trim(), i + 1));

    /* 列幅：列名ベース */

    // ★所属：2文字幅
    if (headerToIndex.has("所属")) {{
      const wTeam = parseInt(getComputedStyle(document.documentElement).getPropertyValue("--w-team")) || 44;
      injectColWidthStyle(headerToIndex.get("所属"), wTeam);
    }}

    // ★選手名：5文字幅 + 左寄せ + 長い名前は自動縮小
    if (headerToIndex.has("選手名")) {{
      const idx = headerToIndex.get("選手名");
      const wName = parseInt(getComputedStyle(document.documentElement).getPropertyValue("--w-name")) || 96;
      injectColWidthStyle(idx, wName, "text-align:left !important;");
      autoShrinkNameCells(table, idx);
    }}

    if (headerToIndex.has("年齢")) injectColWidthStyle(headerToIndex.get("年齢"), 34);
    if (headerToIndex.has("投")) injectColWidthStyle(headerToIndex.get("投"), 34);
    if (headerToIndex.has("打")) injectColWidthStyle(headerToIndex.get("打"), 34);
    if (headerToIndex.has("打席")) injectColWidthStyle(headerToIndex.get("打席"), 64);
    if (headerToIndex.has("得点圏打率")) injectColWidthStyle(headerToIndex.get("得点圏打率"), 72);

    /* 打率以降を同じ幅に */
    if (headerToIndex.has("打率")) {{
      const start = headerToIndex.get("打率");
      const metricWidth = getComputedStyle(document.documentElement)
        .getPropertyValue("--w-metric").trim() || "78px";
      const w = parseInt(metricWidth.replace("px",""), 10) || 78;
      for (let i = start; i <= ths.length; i++) {{
        injectColWidthStyle(i, w);
      }}
    }}

    /* 固定列：選手名のみ */
    freezeColumns(ths, headerToIndex, ["選手名"]);

    /* ソート */
    ths.forEach((th, idx0) => {{
      if (th.dataset.bound === "1") return;
      th.dataset.bound = "1";
      th.dataset.asc = "0";
      th.addEventListener("click", () => {{
        const asc = th.dataset.asc === "1";
        sortTable(table, idx0, asc);
        th.dataset.asc = asc ? "0" : "1";
      }});
    }});
  }}

  bind();
}})();
</script>
</body>
</html>
"""

components.html(full_html, height=820, scrolling=True)





