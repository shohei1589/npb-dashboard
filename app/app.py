import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import math
import streamlit.components.v1 as components
import re
import numpy as np

def format_ip(val):
    """
    投球回の表示を '123 1/3' '123 2/3' 形式に整形する。
    val が 12.1/12.2 方式でも、12.333... 方式でも、outs整数でも対応。
    """
    if val is None:
        return ""

    # 文字列で入ってくるケース（すでに "12 1/3" 等ならそのまま）
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            return ""
        return s

    try:
        # outs（奪アウト数）が整数で入っている場合の救済
        if isinstance(val, (int,)) and val >= 0:
            outs = val
        else:
            x = float(val)

            # よくある 12.1 / 12.2 方式を優先的に解釈
            whole = int(x)
            frac = round(x - whole, 1)

            if abs(frac - 0.1) < 1e-9:
                return f"{whole} 1/3"
            if abs(frac - 0.2) < 1e-9:
                return f"{whole} 2/3"
            if abs(frac - 0.0) < 1e-9:
                return f"{whole}"

            # 12.333... のような“真の小数”の場合：1/3刻みに丸める
            outs = int(round(x * 3))

        innings = outs // 3
        rem = outs % 3
        if rem == 1:
            return f"{innings} 1/3"
        if rem == 2:
            return f"{innings} 2/3"
        return f"{innings}"

    except Exception:
        # 何か変な値でも落とさず文字列化
        return str(val)


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

def ensure_views_updated():
    sql_path = PROJECT_ROOT / "scripts" / "create_views.sql"
    if not sql_path.exists():
        return
    with sqlite3.connect(DB_PATH) as con:
        con.executescript(sql_path.read_text(encoding="utf-8"))
        con.commit()

# 起動時にviewを最新化（デプロイ環境でも反映されるように）
ensure_views_updated()

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

params = st.query_params
is_mobile = str(params.get("mobile", "0")) == "1"

# ===== スマホ判定（画面幅）=====
# 既に is_mobile を作っているならここは不要。無ければ導入する。
is_mobile = st.session_state.get("is_mobile", False)


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


# --- 投手（基本） ---
BASIC_COLS_PIT1 = [
    "選手名","年齢","投","打","防御率","登板","先発","勝利","敗戦","S","HLD",
    "完投","完封","無四球","被打者","投球回","被安打","被本塁打","四球","敬遠",
    "死球","三振","暴投","ボーク","失点","自責点"
]

BASIC_COLS_PIT2 = [
    "選手名","年齢","投","打","防御率","登板","勝利","敗戦","S",
    "完投","完封","無四球","被打者","投球回","被安打","被本塁打","四球","敬遠",
    "死球","三振","暴投","ボーク","失点","自責点"
]

DISPLAY_COLUMNS = {
    ("1軍", "打者成績", "基本"): BASIC_COLS_BAT1,
    ("1軍", "打者成績", "アドバンスド"): ADV_COLS_BAT1,

    ("2軍", "打者成績", "基本"): BASIC_COLS_BAT2,
    ("2軍", "打者成績", "アドバンスド"): ADV_COLS_BAT2,

    # ★追加（投手）
    ("1軍", "投手成績", "基本"): BASIC_COLS_PIT1,
    ("2軍", "投手成績", "基本"): BASIC_COLS_PIT2,

    # 今は投手のアドバンスド未実装なので、同じ列を出す（将来差し替え）
    ("1軍", "投手成績", "アドバンスド"): BASIC_COLS_PIT1,
    ("2軍", "投手成績", "アドバンスド"): BASIC_COLS_PIT2,
}

@st.cache_data
def get_team_pitching_apps(season: int, level: str) -> pd.DataFrame:
    table = "team_pitching_1" if level == "1軍" else "team_pitching_2"
    sql = f"""
    SELECT 年度, 所属, COALESCE(登板,0) AS 登板数
    FROM {table}
    WHERE 年度 = ?
    """
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql(sql, con, params=(season,))


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

def outs_to_ip_str(v) -> str:
    if pd.isna(v):
        return "-"
    try:
        outs = int(float(v))
    except Exception:
        return "-"
    ip = outs // 3
    rem = outs % 3
    if rem == 0:
        return f"{ip}"
    if rem == 1:
        return f"{ip} 1/3"
    return f"{ip} 2/3"

def ip_float_to_fraction_str(v) -> str:
    """
    投球回が小数で来る場合（例: 36.333333 / 13.666667）を
    "36 1/3" / "13 2/3" に変換する。
    すでに文字列ならそのまま返す。
    """
    if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() == ""):
        return "-"

    # すでに "36 1/3" 等ならそのまま
    if isinstance(v, str):
        s = v.strip()
        return s if s else "-"

    try:
        x = float(v)
        outs = int(round(x * 3))  # 1/3刻みに丸め
        ip = outs // 3
        rem = outs % 3
        if rem == 0:
            return f"{ip}"
        elif rem == 1:
            return f"{ip} 1/3"
        else:
            return f"{ip} 2/3"
    except Exception:
        return str(v)

def ip_to_ip_str(v) -> str:
    """
    投球回の表示を統一する:
    - "100 1/3" / "100 2/3" はそのまま
    - "100.1" / "100.2" を "100 1/3" / "100 2/3" に変換
    - 数値 100.1 / 100.2 も同様に変換
    - それ以外は文字列化して返す
    """
    if v is None or pd.isna(v):
        return "-"

    # すでに "100 1/3" 形式ならそのまま
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return "-"
        if re.match(r"^[0-9]+\s+(1/3|2/3)$", s):
            return s
        # "100.1" / "100.2" 文字列
        m = re.match(r"^([0-9]+)\.([12])$", s)
        if m:
            ip = int(m.group(1))
            return f"{ip} 1/3" if m.group(2) == "1" else f"{ip} 2/3"
        return s

    # 数値 100.1 / 100.2 など
    try:
        x = float(v)
        ip = int(x)
        frac = round(x - ip, 1)
        if abs(frac - 0.1) < 1e-9:
            return f"{ip} 1/3"
        if abs(frac - 0.2) < 1e-9:
            return f"{ip} 2/3"
        if abs(frac - 0.0) < 1e-9:
            return f"{ip}"
        # それ以外（変な小数）は落とさず表示
        return str(v)
    except Exception:
        return str(v)

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

def sql_in_clause(items: list[str]) -> tuple[str, list]:
    """
    IN (?, ?, ...) 用のプレースホルダ文字列とparamsを返す
    """
    items = [str(x) for x in items]
    if not items:
        # IN () を避ける（絶対にヒットしない条件にする）
        return "(NULL)", []
    ph = ",".join(["?"] * len(items))
    return f"({ph})", items

@st.cache_data
def get_pitching_1(season: int, team: str) -> pd.DataFrame:
    teams_1gun = list(NPB_TEAMS_1GUN)

    if team == "すべて":
        in_clause, in_params = sql_in_clause(teams_1gun)
        sql = f"""
        SELECT *
        FROM pitching_1_view
        WHERE 年度 = ?
          AND 所属 IN {in_clause}
        ORDER BY 投球回_outs DESC
        """
        params = [season] + in_params

    elif team == "セリーグ":
        in_clause, in_params = sql_in_clause(CENTRAL + BAYSTARS_OLD)
        sql = f"""
        SELECT *
        FROM pitching_1_view
        WHERE 年度 = ?
          AND 所属 IN {in_clause}
        ORDER BY 投球回_outs DESC
        """
        params = [season] + in_params

    elif team == "パリーグ":
        in_clause, in_params = sql_in_clause(PACIFIC)
        sql = f"""
        SELECT *
        FROM pitching_1_view
        WHERE 年度 = ?
          AND 所属 IN {in_clause}
        ORDER BY 投球回_outs DESC
        """
        params = [season] + in_params

    else:
        sql = """
        SELECT *
        FROM pitching_1_view
        WHERE 年度 = ? AND 所属 = ?
        ORDER BY 投球回_outs DESC
        """
        params = [season, team]

    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql(sql, con, params=params)


@st.cache_data
def get_pitching_2(season: int, team: str) -> pd.DataFrame:
    teams_2gun = list(NPB_TEAMS_1GUN + NPB_TEAMS_2GUN_EXTRA)

    if team == "すべて":
        in_clause, in_params = sql_in_clause(teams_2gun)
        sql = f"""
        SELECT *
        FROM pitching_2_view
        WHERE 年度 = ?
          AND 所属 IN {in_clause}
        ORDER BY 投球回_outs DESC
        """
        params = [season] + in_params

    elif team == "イースタン":
        in_clause, in_params = sql_in_clause(EASTERN)
        sql = f"""
        SELECT *
        FROM pitching_2_view
        WHERE 年度 = ?
          AND 所属 IN {in_clause}
        ORDER BY 投球回_outs DESC
        """
        params = [season] + in_params

    elif team == "ウエスタン":
        in_clause, in_params = sql_in_clause(WESTERN)
        sql = f"""
        SELECT *
        FROM pitching_2_view
        WHERE 年度 = ?
          AND 所属 IN {in_clause}
        ORDER BY 投球回_outs DESC
        """
        params = [season] + in_params

    else:
        sql = """
        SELECT *
        FROM pitching_2_view
        WHERE 年度 = ? AND 所属 = ?
        ORDER BY 投球回_outs DESC
        """
        params = [season, team]

    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql(sql, con, params=params)


cols_pitching_1 = [
    "選手名","年齢","投","打","防御率","登板","先発","勝利","敗戦","S","HLD",
    "完投","完封","無四球","被打者","投球回","被安打","被本塁打","四球","敬遠",
    "死球","三振","暴投","ボーク","失点","自責点"
]

cols_pitching_2 = [
    "選手名","年齢","投","打","防御率","登板","勝利","敗戦","S",
    "完投","完封","無四球","被打者","投球回","被安打","被本塁打","四球","敬遠",
    "死球","三振","暴投","ボーク","失点","自責点"
]



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
# 年度は固定レンジ（2005〜2025）
seasons = list(range(2025, 2004, -1))


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

# level別に「広い選択」を定義（ここがブレると所属表示/規定フィルタの初期値が壊れる）
BROAD_TEAMS_1 = ("すべて", "セリーグ", "パリーグ")
BROAD_TEAMS_2 = ("すべて", "イースタン", "ウエスタン")

now_team = st.session_state.get("team")
is_broad_team = (now_team in (BROAD_TEAMS_1 if level == "1軍" else BROAD_TEAMS_2))


# team が変わった瞬間を検知
prev_team = st.session_state.get("_prev_team")
now_team = st.session_state.get("team")
team_changed = (prev_team is not None) and (prev_team != now_team)
st.session_state["_prev_team"] = now_team

# ===== フィルタUI（打者=打席、投手=投球回）=====
if category == "打者成績":
    # teamが「広い選択」に変わったら規定打席へ寄せる
    if "pa_filter" not in st.session_state:
        st.session_state["pa_filter"] = "規定打席" if is_broad_team else "すべて"
    else:
        if team_changed and is_broad_team:
            st.session_state["pa_filter"] = "規定打席"

    pa_options = ["規定打席", "400", "300", "200", "100", "50", "すべて"]
    st.selectbox("打席フィルタ", pa_options, key="pa_filter")

elif category == "投手成績":
    # teamが「広い選択」に変わったら規定投球回へ寄せる
    if "ip_filter" not in st.session_state:
        st.session_state["ip_filter"] = "規定投球回" if is_broad_team else "すべて"
    else:
        if team_changed and is_broad_team:
            st.session_state["ip_filter"] = "規定投球回"

    ip_options = ["規定投球回", "100", "80", "60", "40", "20", "すべて"]
    st.selectbox("投球回フィルタ", ip_options, key="ip_filter")


# ===== 表示（上部ナビに応じて切り替え）=====
st.markdown(f"### {level}・{category}")

if level == "1軍" and category == "打者成績":
    df = get_batting_1(season, team)   # batting_1_view

elif level == "2軍" and category == "打者成績":
    df = get_batting_2(season, team)   # batting_2_view

elif level == "1軍" and category == "投手成績":
    df = get_pitching_1(season, team)  # pitching_1_view（投球回_outs DESC）

elif level == "2軍" and category == "投手成績":
    df = get_pitching_2(season, team)  # pitching_2_view（投球回_outs DESC）

else:
    df = pd.DataFrame()

# ===== スマホだけ表示行数を制限（df作成後が正しい） =====
if is_mobile and (not df.empty):
    st.caption("📱 スマホ表示：上位のみ表示")
    n_rows = st.selectbox("表示人数", options=[50, 100, 200, "全件"], index=0, key="mobile_n_rows")

    if n_rows != "全件":
        if category == "打者成績" and "打席" in df.columns:
            df = df.sort_values("打席", ascending=False).head(int(n_rows))
        elif category == "投手成績" and "投球回_outs" in df.columns:
            df = df.sort_values("投球回_outs", ascending=False).head(int(n_rows))
        else:
            df = df.head(int(n_rows))

# ---- 投手：投球回表示（"100 1/3" / "100 2/3"）を確実にする ----
if category == "投手成績" and (not df.empty):

    # まず「投球回(小数)」から outs を作って分数表示にする（※こちらを優先）
    # 例: 109.666667 -> 329 outs -> "109 2/3"
    if "投球回" in df.columns:
        ip_num = pd.to_numeric(df["投球回"], errors="coerce")
        outs_from_ip = (ip_num * 3).round().astype("Int64")  # 1/3刻み前提で丸め
        df["投球回"] = outs_from_ip.apply(outs_to_ip_str)

    # 投球回が無い/変換不能の保険：outs列があるならそれを使う
    if ("投球回" not in df.columns) and ("投球回_outs" in df.columns):
        df["投球回_outs"] = pd.to_numeric(df["投球回_outs"], errors="coerce").fillna(0).astype(int)
        df["投球回"] = df["投球回_outs"].apply(outs_to_ip_str)





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



# 所属列は「すべて/リーグ」だけで表示
show_team_col = team in (("すべて", "セリーグ", "パリーグ") if level == "1軍" else ("すべて", "イースタン", "ウエスタン"))



# ---- 2) 年度/所属は “ここでは落とさない” ----
# 理由：投球回フィルタ（規定投球回）で 年度/所属 を使って merge するため
# ※表示列の整形（列出し分け）の段階で自然に落ちるのでここでは触らない


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

# ---- 投手：投球回フィルタ ----
if category == "投手成績" and (not df.empty) and ("投球回_outs" in df.columns):
    ip_filter = st.session_state.get("ip_filter", "すべて")

    df["投球回_outs"] = pd.to_numeric(df["投球回_outs"], errors="coerce").fillna(0).astype(int)

    if ip_filter != "すべて":
        if ip_filter in ["20", "40", "60", "80", "100"]:
            thr_outs = int(float(ip_filter) * 3)
            df = df[df["投球回_outs"] >= thr_outs].copy()

        elif ip_filter == "規定投球回":
            apps_df = get_team_pitching_apps(season, level)  # 年度, 所属, 登板数
            apps_df["登板数"] = pd.to_numeric(apps_df["登板数"], errors="coerce").fillna(0)

            # 規定係数：1軍=1.0、2軍=0.8
            factor = 1.0 if level == "1軍" else 0.8

            # 所属で結合して、行ごとに規定を計算
            df = df.merge(apps_df, on=["年度", "所属"], how="left")
            df["登板数"] = pd.to_numeric(df["登板数"], errors="coerce").fillna(0)
            df["規定投球回_outs"] = (df["登板数"] * factor * 3.0).round().astype(int)

            df = df[df["投球回_outs"] >= df["規定投球回_outs"]].copy()

# ---- 所属略称化は「規定投球回のmerge後」に行う（merge一致が壊れるため）----
if show_team_col and ("所属" in df.columns):
    df["所属"] = df["所属"].astype(str).str.strip()
    df["所属"] = df["所属"].map(TEAM_ABBR).fillna(df["所属"])

# ---- 投手：投球回の表示を必ず "xx 1/3" / "xx 2/3" に統一する ----
if category == "投手成績" and (not df.empty):

    # ★最優先：outs があるなら必ず outs から作り直す（端数情報の源泉）
    if "投球回_outs" in df.columns:
        df["投球回_outs"] = pd.to_numeric(df["投球回_outs"], errors="coerce").fillna(0).astype(int)
        df["投球回"] = df["投球回_outs"].apply(outs_to_ip_str)

    # outs が無い場合だけ、投球回（文字列/小数）を整形する（保険）
    elif "投球回" in df.columns:
        df["投球回"] = df["投球回"].apply(ip_to_ip_str)

# ---- 5) デフォルトソート（打者：打席降順） ----
if level in ("1軍", "2軍") and category == "打者成績" and "打席" in df.columns:
    df = df.sort_values("打席", ascending=False, na_position="last")

# ---- 6) 列出し分け（基本/アドバンスド） ----
if level in ("1軍", "2軍") and category in ("打者成績", "投手成績"):
    mode = st.session_state.get("show_mode", "基本")
    key = (level, category, mode)
    if key in DISPLAY_COLUMNS:
        cols = DISPLAY_COLUMNS[key]

        if (not show_team_col) and ("所属" in cols):
            cols = [c for c in cols if c != "所属"]

        # ★「すべて/リーグ」の時は “所属” を先頭に表示（略称化済みの所属を見せる）
        if show_team_col and ("所属" in df.columns) and ("所属" not in cols):
            cols = ["所属"] + cols

        # 得点圏打率は 1軍のみ 2016年未満で落とす（2軍はそもそも列セットに無い）
        if level == "1軍" and season < 2016 and "得点圏打率" in cols:
            cols = [c for c in cols if c != "得点圏打率"]

        cols_exist = [c for c in cols if c in df.columns]
        df = df[cols_exist]

if (not show_team_col) and ("所属" in df.columns):
    df = df.drop(columns=["所属"], errors="ignore")


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
    "盗塁", "盗塁死", "併殺打","登板","先発","勝利","敗戦","S","HLD","完投","完封","無四球","被打者","被安打","被本塁打","暴投","ボーク","失点","自責点"

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

# 投球回を左寄せ＆省略（…）されないようにする（JSに依存しない）
if "投球回" in df.columns:
    ip_idx = list(df.columns).index("投球回") + 1  # nth-childは1始まり
    table_styles.append({
        "selector": f"thead th:nth-child({ip_idx})",
        "props": [
            ("text-align", "left !important"),
            ("min-width", "90px !important"),
            ("max-width", "90px !important"),
            ("width", "90px !important"),
        ],
    })
    table_styles.append({
        "selector": f"tbody td:nth-child({ip_idx})",
        "props": [
            ("text-align", "left !important"),
            ("min-width", "90px !important"),
            ("max-width", "90px !important"),
            ("width", "90px !important"),
            ("overflow", "visible !important"),
            ("text-overflow", "clip !important"),
        ],
    })

if "投球回" in fmt:
    del fmt["投球回"]

# 5) Styler：順番を固定（apply → format → styles）
styler = df.style
styler = styler.apply(apply_heatmap, axis=None)  # 背景色（数値が必要）
styler = styler.format(fmt)                      # 表示形式
styler = styler.set_table_styles(table_styles)   # CSS

# 投球回は左寄せ（Styler側でも明示しておく）
if "投球回" in df.columns:
    styler = styler.set_properties(subset=["投球回"], **{"text-align": "left !important"})


# --- 投球回列の位置（HTML nth-child用）を特定 ---
ip_col_idx = None
if "投球回" in df.columns:
    ip_col_idx = list(df.columns).index("投球回") + 1  # nth-childは1始まり

# --- 年齢/防御率列の位置（HTML nth-child用）を特定 ---
age_col_idx = None
era_col_idx = None
if "年齢" in df.columns:
    age_col_idx = list(df.columns).index("年齢") + 1
if "防御率" in df.columns:
    era_col_idx = list(df.columns).index("防御率") + 1

age_col_css = ""
if age_col_idx is not None:
    age_col_css = f"""
  /* 年齢列：狭くする */
  thead th:nth-child({age_col_idx}),
  tbody td:nth-child({age_col_idx}) {{
    min-width: 56px !important;
    width: 56px !important;
    max-width: 56px !important;
  }}
"""

era_col_css = ""
if era_col_idx is not None:
    era_col_css = f"""
  /* 防御率列：広くする */
  thead th:nth-child({era_col_idx}),
  tbody td:nth-child({era_col_idx}) {{
    min-width: 92px !important;
    width: 92px !important;
    max-width: 92px !important;
  }}
"""


# --- 「選手名列」を特定（JS側で左寄せクラス付与に使う）---
name_col = None
for cand in ["選手名", "名前", "選手"]:
    if cand in df.columns:
        name_col = cand
        break
if name_col is None and len(df.columns) > 0:
    name_col = df.columns[0]
if name_col is None:
    name_col = ""

# --- 数値列（指標列）を特定（JS側で固定幅クラス付与に使う）---
metric_cols = []
for c in df.columns:
    try:
        if pd.api.types.is_numeric_dtype(df[c]):
            metric_cols.append(str(c))
    except Exception:
        pass

# JSに渡す用（Python list をそのまま JS Array として埋め込む）
metric_cols_js = repr(metric_cols)

# 投球回列専用CSS（左寄せ＋幅確保＋省略しない）
ip_col_css = ""
if ip_col_idx is not None:
    ip_col_css = f"""
  /* 投球回列だけ：左寄せ + 省略しない + 幅を確保 */
  thead th:nth-child({ip_col_idx}) {{
    text-align: left !important;
    min-width: 90px !important;
    width: 90px !important;
    max-width: 90px !important;
  }}
  tbody td:nth-child({ip_col_idx}) {{
    text-align: left !important;
    min-width: 90px !important;
    width: 90px !important;
    max-width: 90px !important;
    overflow: visible !important;
    text-overflow: clip !important;
  }}
"""

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

    /* 指標列の統一幅（PC） */
    --w-metric: 78px;

    /* 選手名列：5文字程度 */
    --w-name: 96px;
  }}

  /* 外枠（内部スクロール） */
  .tbl-wrap {{
    width: 100%;
    overflow: auto;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    background: white;
    max-height: 720px;   /* PC */
  }}

  /* モバイルは画面に合わせる */
  @media (max-width: 768px) {{
    .tbl-wrap {{
      max-height: 65vh;
    }}
  }}

  table {{
    border-collapse: separate;
    border-spacing: 0;
    width: max-content;
    min-width: 100%;
    color: var(--text);
  }}

  thead th {{
    position: sticky;
    top: 0;
    z-index: 2;
    background: #f9fafb;
    border-bottom: 1px solid var(--border);
    border-right: 1px solid var(--border2);
    padding: var(--th-pad-y) var(--th-pad-x);
    font-size: var(--th-font);
    font-weight: 700;
    white-space: nowrap;
    text-align: center;
  }}

  tbody td {{
    border: 1px solid var(--border2);
    padding: var(--td-pad-y) var(--td-pad-x);
    font-size: var(--td-font);
    white-space: nowrap;
    text-align: center;
    overflow: hidden;
    text-overflow: ellipsis;
    background: #ffffff;
  }}

  /* 指標列は固定幅（数値列） */
  tbody td.metric, thead th.metric {{
    min-width: var(--w-metric);
    width: var(--w-metric);
    max-width: var(--w-metric);
  }}

  /* 選手名列は固定幅 */
  tbody td.name, thead th.name {{
    min-width: var(--w-name);
    width: var(--w-name);
    max-width: var(--w-name);
    text-align: left;
  }}

  /* ★選手名が溢れるセルだけフォントを小さくする */
  tbody td.name.shrink {{
    font-size: calc(var(--td-font) - 2px);
  }}
  @media (max-width: 768px) {{
    tbody td.name.shrink {{
      font-size: calc(var(--td-font) - 1px);
    }}
  }}

  /* ★選手名列を固定表示（左端固定） */
  thead th.name.sticky {{
    position: sticky !important;
    left: 0px !important;
    z-index: 8 !important;
    background: #f9fafb !important;
    box-shadow: 2px 0 0 rgba(0,0,0,0.06);
  }}
  tbody td.name.sticky {{
    position: sticky !important;
    left: 0px !important;
    z-index: 6 !important;
    background: #ffffff !important;
    box-shadow: 2px 0 0 rgba(0,0,0,0.06);
  }}

  /* 右端の余白カット */
  thead th:last-child, tbody td:last-child {{
    border-right: 0;
  }}

  /* ★投球回・年齢・防御率の列幅等（Pythonで生成したCSSを差し込み） */
  {ip_col_css}
  {age_col_css}
  {era_col_css}

  /* スマホでは少し詰める */
  @media (max-width: 768px) {{
    :root {{
      --th-font: 12px;
      --td-font: 13px;
      --th-pad-y: 7px;
      --th-pad-x: 8px;
      --td-pad-y: 6px;
      --td-pad-x: 8px;
      --w-metric: 72px;
      --w-name: 90px;
    }}
  }}
</style>
</head>
<body>
<div class="tbl-wrap">
{html_table}
</div>

<script>
(() => {{
  const wrap = document.querySelector(".tbl-wrap");
  const table = wrap?.querySelector("table");
  if (!table) return;

  // th/td にクラス付与（名前列・指標列）
  const ths = table.querySelectorAll("thead th");
  const metricSet = new Set({metric_cols_js});

  ths.forEach((th, idx0) => {{
    const colName = th.textContent?.trim() || "";
    if (colName === "{name_col}") {{
      th.classList.add("name");
      table.querySelectorAll(`tbody tr`).forEach(tr => {{
        const td = tr.children[idx0];
        if (td) td.classList.add("name");
      }});
    }} else if (metricSet.has(colName)) {{
      th.classList.add("metric");
      table.querySelectorAll(`tbody tr`).forEach(tr => {{
        const td = tr.children[idx0];
        if (td) td.classList.add("metric");
      }});
    }}
  }});

  // ★選手名セルがはみ出していたらフォント縮小
  function applyNameShrink() {{
    table.querySelectorAll("tbody td.name").forEach(td => {{
      td.classList.remove("shrink");
      if (td.scrollWidth > td.clientWidth + 1) {{
        td.classList.add("shrink");
      }}
    }});
  }}
  applyNameShrink();

  // ★選手名列を左端に固定（横スクロールしても動かない）
  function applyStickyName() {{
    const nameTh = table.querySelector("thead th.name");
    if (!nameTh) return;
    nameTh.classList.add("sticky");
    table.querySelectorAll("tbody td.name").forEach(td => td.classList.add("sticky"));
  }}
  applyStickyName();

  // ソート（クリックで昇順↔降順）
  function getCellValue(tr, idx) {{
    const td = tr.children[idx];
    if (!td) return "";
    return td.textContent.trim();
  }}

  function parseVal(v) {{
    const n = Number(v.replace(/,/g, ""));
    if (!Number.isNaN(n)) return n;
    return v;
  }}

  function sortTable(tbl, colIdx, asc) {{
    const tbody = tbl.tBodies[0];
    const rows = Array.from(tbody.rows);
    rows.sort((a, b) => {{
      const va = parseVal(getCellValue(a, colIdx));
      const vb = parseVal(getCellValue(b, colIdx));
      if (typeof va === "number" && typeof vb === "number") {{
        return asc ? va - vb : vb - va;
      }}
      return asc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
    }});
    rows.forEach(r => tbody.appendChild(r));

    // ソート後に再適用
    applyNameShrink();
    applyStickyName();
  }}

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
}})();
</script>
</body>
</html>
"""


row_h = 34 if is_mobile else 0
if is_mobile:
    # ヘッダー1行 + データ行 + 余白
    est_h = int((len(df) + 1) * row_h + 220)
    est_h = min(est_h, 2400)  # 伸びすぎ防止（必要なら調整）
    components.html(full_html, height=est_h, scrolling=False)
else:
    components.html(full_html, height=820, scrolling=False)