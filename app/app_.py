from pathlib import Path
from typing import List



CUSTOM_CSS = r"""
<style>
/* Font */
html, body, [class*="css"], .stApp {
  font-family: "Meiryo UI", Meiryo, "Yu Gothic UI", "Hiragino Kaku Gothic ProN", Arial, sans-serif !important;
}

/* Table wrapper */
.npb-table-wrap{
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  background: #ffffff;
  box-shadow: 0 8px 18px rgba(0,0,0,0.06);
  overflow: auto;
  max-height: 72vh;
}

/* Table */
.npb-table{
  border-collapse: separate;
  border-spacing: 0;
  width: max-content;
  min-width: 100%;
  font-size: 13px;
}
.npb-table th, .npb-table td{
  border-right: 1px solid #e5e7eb;
  border-bottom: 1px solid #e5e7eb;
  padding: 8px 10px;
  white-space: nowrap;
  text-align: center;
}
.npb-table thead th{
  position: sticky;
  top: 0;
  background: #f9fafb;     /* non-transparent */
  z-index: 5;
  font-weight: 700;
}
.npb-table tbody tr:nth-child(even) td{
  background: #fcfcfd;
}

/* Double separator column (between 年齢 and 登板/試合 etc) */
.npb-table th.sep, .npb-table td.sep{
  border-right: 4px double #9ca3af !important;
}

/* Sticky first 4 columns (freeze pane: after 4th column) */
.npb-table th.sticky, .npb-table td.sticky{
  position: sticky;
  background: #ffffff;
  z-index: 4;
}
.npb-table thead th.sticky{
  background: #f9fafb;
  z-index: 6;
}
.npb-table th.sticky.col1, .npb-table td.sticky.col1{ left: 0px; }
.npb-table th.sticky.col2, .npb-table td.sticky.col2{ left: 72px; }
.npb-table th.sticky.col3, .npb-table td.sticky.col3{ left: 232px; }
.npb-table th.sticky.col4, .npb-table td.sticky.col4{ left: 296px; }

/* Left align for innings only */
.npb-table td.left{
  text-align: left !important;
}

/* Make numeric columns a bit narrower */
.npb-table col.numcol{ width: 72px; }
.npb-table col.wide{ width: 160px; }
.npb-table col.narrow{ width: 64px; }
.npb-table col.tiny{ width: 72px; }

/* Remove Streamlit attachment footer spacing if any */
footer {visibility: hidden;}
</style>
"""



# -*- coding: utf-8 -*-
import math
import pandas as pd
import numpy as np
import hashlib
import streamlit as st
import streamlit.components.v1 as components

APP_TITLE = "NPB ダッシュボード"

NPB_TEAMS_12 = [
    "ソフトバンク","日本ハム","オリックス","楽天","ロッテ","西武",
    "巨人","阪神","中日","広島","ヤクルト","DeNA","横浜",
]
NPB_TEAMS_EXTRA = ["オイシックス", "ハヤテ"]

def excel_col_to_idx(col: str) -> int:
    """Convert Excel column letters (e.g., 'A', 'AI', 'BG') to 0-based index."""
    col = col.strip().upper()
    n = 0
    for ch in col:
        if not ("A" <= ch <= "Z"):
            continue
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def add_top_batting_derived_cols(top_b: pd.DataFrame) -> pd.DataFrame:
    """1軍打撃に必要な列（打率/出塁率/長打率/OPS/wOBA/BB%）をExcel列位置ベースで追加し、数値化しておく。"""
    df = top_b.copy()
    mapping = {
        "打率": "H",
        "出塁率": "AI",
        "長打率": "AJ",
        "OPS": "AK",
        "wOBA": "BG",
        "BB%_raw": "BI",
    }
    for name, col in mapping.items():
        idx = excel_col_to_idx(col)
        if 0 <= idx < df.shape[1] and name not in df.columns:
            df[name] = df.iloc[:, idx]

    # 数値化（"0.312" / "31.2%" / "31.2" / "-" など混在を許容）
    def to_num(s: pd.Series) -> pd.Series:
        if s.dtype == object:
            t = s.astype(str).str.strip()
            t = t.replace({"-": np.nan, "－": np.nan, "": np.nan})
            t = t.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            return pd.to_numeric(t, errors="coerce")
        return pd.to_numeric(s, errors="coerce")

    for c in ["打率", "出塁率", "長打率", "OPS", "wOBA"]:
        if c in df.columns:
            df[c] = to_num(df[c])

    if "BB%" not in df.columns and "BB%_raw" in df.columns:  # BI列由来（プロトタイプ）
        s = to_num(df["BB%_raw"])
        med = float(s.dropna().median()) if s.notna().any() else np.nan
        # 0-1帯なら「割合」とみなして%化
        if not np.isnan(med) and med <= 1.0:
            df["BB%"] = s * 100.0
        else:
            df["BB%"] = s

    return df


def add_farm_batting_derived_cols(farm_b: pd.DataFrame) -> pd.DataFrame:
    """2軍打撃に必要な列をできる範囲で追加。

    基本的には1軍と同じ列レター（H, AI-AK, BG, BI）を優先して読みます。
    2軍側で列配置が異なる場合でも落ちないよう、存在する列だけ追加します。
    """
    if farm_b is None:
        return farm_b
    df = farm_b.copy()

    mapping = {
        "打率": "H",
        "出塁率": "AI",
        "長打率": "AJ",
        "OPS": "AK",
        "wOBA": "BG",
        "BB%_raw": "BI",
    }
    for name, col in mapping.items():
        idx = excel_col_to_idx(col)
        if 0 <= idx < df.shape[1] and name not in df.columns:
            df[name] = df.iloc[:, idx]

    # 数値化（"0.312" / "31.2%" / "31.2" / "-" など混在を許容）
    def to_num(s: pd.Series) -> pd.Series:
        if s.dtype == object:
            t = s.astype(str).str.strip()
            t = t.replace({"-": np.nan, "－": np.nan, "": np.nan})
            t = t.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            return pd.to_numeric(t, errors="coerce")
        return pd.to_numeric(s, errors="coerce")

    for c in ["打率", "出塁率", "長打率", "OPS", "wOBA"]:
        if c in df.columns:
            df[c] = to_num(df[c])

    if "BB%" not in df.columns and "BB%_raw" in df.columns:
        s = to_num(df["BB%_raw"])
        med = float(s.dropna().median()) if s.notna().any() else np.nan
        if not np.isnan(med) and med <= 1.0:
            df["BB%"] = s * 100.0
        else:
            df["BB%"] = s

    return df


def fmt_three_no_leading_zero(x):
    """Format as .000 (no leading zero) when 0<=x<1, else 0.000/1.234 style."""
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        v = float(x)
    except Exception:
        return str(x)
    s = f"{v:.3f}"
    if 0 <= v < 1 and s.startswith("0"):
        s = s[1:]
    return s


def normalize_jersey(j) -> str:
    """Normalize jersey number for display/sort.

    Excel sometimes returns numeric jerseys as float (e.g., 77.0). We convert
    integer-like floats to an int string. If a jersey is already a string with
    leading zeros (e.g., "00", "001"), preserve it.
    """
    if j is None or (isinstance(j, float) and np.isnan(j)):
        return ""
    # pandas NA
    try:
        if pd.isna(j):
            return ""
    except Exception:
        pass

    # numeric -> int if whole
    if isinstance(j, (int, np.integer)):
        return str(int(j))
    if isinstance(j, (float, np.floating)):
        if np.isfinite(j) and float(j).is_integer():
            return str(int(j))
        return str(j)

    s = str(j).strip()
    if s == "":
        return ""

    # If already zero-padded digits, keep as-is
    if s.isdigit() and len(s) > 1 and s.startswith("0"):
        return s

    # Try coerce strings like "77.0" -> "77"
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s

def ip_to_float(ip_str: str) -> float:
    if ip_str is None or (isinstance(ip_str, float) and np.isnan(ip_str)):
        return 0.0
    s = str(ip_str).strip()
    if s == "":
        return 0.0
    import re
    m = re.match(r"^(-?\d+)\s+(1/3|2/3)$", s)
    if m:
        ip = int(m.group(1))
        frac = 1/3 if m.group(2) == "1/3" else 2/3
        return ip + frac
    try:
        return float(s)
    except Exception:
        return 0.0

def fmt_ip(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    v = float(v)
    n = int(math.floor(v + 1e-9))
    frac = v - n
    if abs(frac) < 1e-6:
        return f"{n}"
    if abs(frac - 1/3) < 1e-3:
        return f"{n} 1/3"
    if abs(frac - 2/3) < 1e-3:
        return f"{n} 2/3"
    return f"{v:.1f}"

def pct(n, d):
    """Percent (0-100) with safe divide. Accepts scalars or pandas Series/ndarray."""
    n_s = pd.to_numeric(n, errors="coerce")
    d_s = pd.to_numeric(d, errors="coerce")
    return np.where((d_s > 0) & (~pd.isna(d_s)), 100.0 * (n_s / d_s), np.nan)


def percent_from_excel(s: pd.Series) -> pd.Series:
    """Excel由来の%列を 0-100 スケールに揃える。
    - 0.052 のような小数(=5.2%)で入ってくるケースを想定し、代表値が <= 1 なら *100 する。
    - 文字列や '-' はそのまま NaN 扱いにして、表示側で '-' に統一する。
    """
    x = pd.to_numeric(s, errors="coerce")
    # 代表値で判定（全欠損のときはそのまま）
    if x.notna().any():
        med = float(x.dropna().median())
        if abs(med) <= 1.0:
            x = x * 100.0
    return x

def pick_jersey(row):
    j2 = normalize_jersey(row.get("背番号_2", np.nan))
    if j2 != "":
        return j2
    return normalize_jersey(row.get("背番号", np.nan))

def jersey_sort_key(j):
    if j is None or (isinstance(j, float) and np.isnan(j)):
        return (9, 9999)
    s = normalize_jersey(j)
    if s == "":
        return (9, 9999)
    if s == "0":
        return (0, 0)
    if s == "00":
        return (1, 0)
    if s.isdigit():
        if len(s) == 1:
            return (2, int(s))
        if len(s) == 2:
            return (3, int(s))
        return (4, int(s))
    return (9, 9999)

def filter_team_year_options(team: str):
    if team in ["オイシックス","ハヤテ"]:
        return list(range(2024, 2026))
    return list(range(2005, 2026))

def pick_col_by_patterns(df: pd.DataFrame, patterns: list[str]) -> str | None:
    """Return first column name that contains any of the given substrings."""
    for c in df.columns:
        sc = str(c)
        for p in patterns:
            if p in sc:
                return c
    return None

@st.cache_data(show_spinner=False)
def load_excel(excel_path: str):
    aff = pd.read_excel(excel_path, sheet_name="選手所属", header=1)
    aff["所属_norm"] = aff["所属"].astype(str)
    aff.loc[(aff["年度"] <= 2011) & (aff["所属_norm"] == "横浜"), "所属_norm"] = "DeNA"

    # シート名が違うExcelにも対応できるように候補を持たせる
    top_p = read_sheet_any(excel_path, ["一軍基本_投球", "一軍基本_投手", "1軍基本_投球", "1軍基本_投手"])
    farm_p = read_sheet_any(excel_path, ["二軍基本_投球", "二軍基本_投手", "2軍基本_投球", "2軍基本_投手", "ファーム基本_投球"])
    # --- Excel列(P/Q/R)をエイリアスとして保持（表示順の追加用） ---
    # ※列名が変わっても「列位置」で取れるようにする（P=16列目, Q=17列目, R=18列目: 0-indexで15/16/17）
    def _alias_pqr(df: pd.DataFrame, tag: str) -> pd.DataFrame:
        global PQR_DISPLAY
        idx_map = {"P": 15, "Q": 16, "R": 17}
        for letter, idx in idx_map.items():
            if idx < df.shape[1]:
                PQR_DISPLAY.setdefault(tag, {})[letter] = str(df.columns[idx])
                df[f"__{tag}_{letter}"] = df.iloc[:, idx]
        return df

    top_p = _alias_pqr(top_p, "TOPP")
    farm_p = _alias_pqr(farm_p, "FARMP")

    # --- 二軍投手: 一部指標は列名ではなく Excel の列位置で明示的に拾う ---
    # ユーザー指定: S=K列, 被安打=R列, 被本塁打=S列, HR%=AF列（Hは無し）
    def _alias_excel_cols(df: pd.DataFrame, tag: str, mapping: dict) -> pd.DataFrame:
        """mapping: {new_col: 'K' or idx(int)}"""
        for new_col, colref in mapping.items():
            try:
                idx = excel_col_to_idx(colref) if isinstance(colref, str) else int(colref)
            except Exception:
                continue
            if 0 <= idx < df.shape[1]:
                df[new_col] = df.iloc[:, idx]
        return df

    farm_p = _alias_excel_cols(
        farm_p,
        "FARMP",
        {
            "__FARMP_S_K": "K",      # セーブ（S）
            "__FARMP_HA_R": "R",     # 被安打
            "__FARMP_HR_S": "S",     # 被本塁打
            "__FARMP_HR_PCT_AF": "AF"  # HR%
        },
    )

    # HR% は「AF列」として取得しているが、Excelの構造（先頭の空列/結合セル等）により
    # pandasの列位置がズレて AF が意図した列を指さないケースがある。
    # その場合でも HR% を正しく表示できるよう、列名に "HR%"/"HR％" を含む列が存在すれば
    # それを優先して採用する（既存ロジックはフォールバックとして温存）。
    if farm_p is not None and not farm_p.empty:
        hr_hdr = pick_col_by_patterns(farm_p, ["HR%", "HR％"])
        if hr_hdr is not None:
            farm_p["__FARMP_HR_PCT_AF"] = farm_p[hr_hdr]

    top_b = read_sheet_any(excel_path, ["一軍基本_打撃", "一軍基本_打者", "1軍基本_打撃", "1軍基本_打者"])
    top_b = add_top_batting_derived_cols(top_b)
    farm_b = read_sheet_any(excel_path, ["二軍基本_打撃", "二軍基本_打者", "2軍基本_打撃", "2軍基本_打者", "ファーム基本_打撃"])
    farm_b = add_farm_batting_derived_cols(farm_b)

    if "リーグ" in top_p.columns:
        top_p = top_p[top_p["リーグ"].isna()]
    if "リーグ" in farm_p.columns:
        farm_p = farm_p[farm_p["リーグ"].isna()]

    return aff, top_p, farm_p, top_b, farm_b

def roster_for(aff, year, team):
    team_norm = "DeNA" if team == "DeNA" else team
    sub = aff[(aff["年度"] == year) & (aff["所属_norm"] == team_norm)].copy()
    sub["背番号_use"] = sub.apply(pick_jersey, axis=1)
    sub["投打"] = sub["投"].astype(str).str.strip() + "/" + sub["打"].astype(str).str.strip()
    sub["__jkey"] = sub["背番号_use"].apply(jersey_sort_key)
    sub = sub.sort_values("__jkey")
    return sub

def merge_roster_stats(roster, stats, year, team):
    if stats is None or stats.empty:
        return roster.copy()
    s = stats.copy()
    if "所属" in s.columns:
        s["所属_norm"] = s["所属"].astype(str)
        s.loc[(s["年度"] <= 2011) & (s["所属_norm"] == "横浜"), "所属_norm"] = "DeNA"
    else:
        s["所属_norm"] = ""
    s = s[(s["年度"] == year) & (s["所属_norm"] == ("DeNA" if team == "DeNA" else team))].copy()
    key_cols = [c for c in ["選手ID","選手名","年度","所属_norm","年齢","投","打"] if c in s.columns]

    # Excel 由来の列は object になりがちなので、(キー列以外は) できるだけ数値化してから集計する。
    # "-" は欠損扱い (NaN) にし、"%" や "," も除去。
    for _c in s.columns:
        if _c in key_cols:
            continue
        if s[_c].dtype == object:
            _tmp = s[_c].astype(str).str.strip()
            _tmp = _tmp.replace({"-": np.nan, "": np.nan})
            _tmp = _tmp.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            s[_c] = pd.to_numeric(_tmp, errors="coerce")
    num_cols = [c for c in s.columns if c not in key_cols and pd.api.types.is_numeric_dtype(s[c])]
    if "選手ID" in s.columns and len(num_cols) > 0:
        s_sum = s.groupby("選手ID", as_index=False)[num_cols].sum()
        meta = s.groupby("選手ID", as_index=False).first()[["選手ID","選手名","年齢","投","打"]]
        s = meta.merge(s_sum, on="選手ID", how="left")

    out = roster.merge(s, on="選手ID", how="left", suffixes=("", "_stat"))
    return out

def pitcher_table(df):
    d = df.copy()
    # 2軍投手は列名が揃っていないため、Excel列位置で拾った __FARMP_* の別名列が入る
    is_farm = any(c.startswith("__FARMP_") for c in d.columns) and not any(c.startswith("__TOPP_") for c in d.columns)
    g_col = "試合" if "試合" in d.columns else ("登板" if "登板" in d.columns else None)
    ip_col = "回数" if "回数" in d.columns else ("投球回" if "投球回" in d.columns else None)
    bf_col = "被打者" if "被打者" in d.columns else ("打者" if "打者" in d.columns else None)

    if ip_col:
        d["__ip_float"] = d[ip_col].apply(ip_to_float)
    else:
        d["__ip_float"] = 0.0

    # hide players with no pitching appearances (登板/試合=0 and 回数=0)
    if g_col and g_col in d.columns:
        gnum = pd.to_numeric(d[g_col], errors="coerce").fillna(0)
    else:
        gnum = pd.Series([0]*len(d), index=d.index)
    d = d[(gnum > 0) | (d["__ip_float"] > 0)].copy()

    if bf_col in d.columns:
        d["K%"] = pct(d["三振"], d[bf_col])
        d["BB%"] = pct(d["四球"], d[bf_col])
        d["HR%"] = pct(d["被本塁打"], d[bf_col]) if "被本塁打" in d.columns else np.nan
        d["K-BB%"] = d["K%"] - d["BB%"]
    else:
        d["K%"] = d["BB%"] = d["HR%"] = d["K-BB%"] = np.nan

    # 2軍投手: 指定列で上書き（列名の揺れ/重複を回避）
    if is_farm:
        # 被安打=R列, 被本塁打=S列（ユーザー指定）
        if "__FARMP_HA_R" in d.columns:
            d["被安打"] = d["__FARMP_HA_R"]
        if "__FARMP_HR_S" in d.columns:
            d["被本塁打"] = d["__FARMP_HR_S"]
        # HR%（2軍投手）: Excel列AF。
        # v34では __FARMP_HR_PCT_AF を作っていますが、過去の互換で __FARMP_AF_AS_HR_PCT も扱います。
        if "__FARMP_HR_PCT_AF" in d.columns:
            d["HR%"] = d["__FARMP_HR_PCT_AF"]
        elif "__FARMP_AF_AS_HR_PCT" in d.columns:
            d["HR%"] = d["__FARMP_AF_AS_HR_PCT"]

        # HR% が 0-1 形式で入っている場合は 0-100 に変換
        if "HR%" in d.columns:
            hr_num = pd.to_numeric(d["HR%"], errors="coerce")
            if hr_num.notna().any():
                med = float(hr_num.dropna().median())
                if med <= 1.0:
                    d["HR%"] = hr_num * 100.0

    cols = ["背番号_use","選手名","年齢","投打"]
    if g_col:
        cols += [g_col]
    # 勝敗/SV まわり：シートによって "S" と "セーブ" が両方あることがあり、
    # さらに farm 投手では Q 列を "S" として表示するため、重複列名が起きやすい。
    # ここでは (1) "S" がある場合は "セーブ" を追加しない。
    # さらに farm 投手はユーザー指定の "S=K列" を優先するため、元シートの "S" は表示しない。
    farm_s_override = is_farm and ("__FARMP_S_K" in d.columns)
    for c in ["勝利", "敗戦", "S", "セーブ"]:
        if farm_s_override and c in {"S", "セーブ"}:
            continue
        if c == "セーブ" and "S" in d.columns:
            continue
        if c in d.columns:
            cols.append(c)
    # 敗戦と回数の間の追加列
    # 1軍: Excel列(Q→P→R) を S/H/HP として表示
    # 2軍: ユーザー指定により S=K列、H無し（=表示しない）。HPは二軍では表示しない。
    if is_farm:
        # 2軍: S=K列（ユーザー指定）
        extra_cols = ["__FARMP_S_K"]  # S only
    else:
        extra_cols = ["__TOPP_Q", "__TOPP_P", "__TOPP_R"]

    if "敗戦" in cols:
        ins = cols.index("敗戦") + 1
        for c in extra_cols:
            if c in d.columns and c not in cols:
                cols.insert(ins, c)
                ins += 1
    else:
        for c in extra_cols:
            if c in d.columns and c not in cols:
                cols.append(c)

    if ip_col:
        cols.append(ip_col)
    for c in ["被安打","被本塁打","三振","四球","死球","防御率"]:
        if c in d.columns:
            cols.append(c)
    cols += ["K%","BB%","K-BB%","HR%"]

    out = d[cols].copy()
    rename = {"背番号_use":"背番号"}
    if g_col == "試合":
        rename["試合"] = "登板"
    if ip_col == "投球回":
        rename["投球回"] = "回数"
    # Q列を "S" として表示するため、"セーブ" は "SV" にして衝突を回避
    if "セーブ" in out.columns:
        rename["セーブ"] = "SV"
    if ip_col == "回数":
        rename["回数"] = "回数"
    out = out.rename(columns=rename)

    # Q列を "S" として表示する場合、もともと "S" 列があるシートでは列名が重複する。
    # 重複すると df["S"] が DataFrame になり、後続処理で落ちるため退避しておく。
    if "S" in out.columns and any(c in out.columns for c in ("__TOPP_Q", "__FARMP_Q", "Q")):
        out = out.rename(columns={"S": "S(旧)"})

    # 追加したExcel列(Q/P/R)の列名を元の表示名に戻す
    for tag in ("TOPP", "FARMP"):
        for letter in ("Q", "P", "R"):
            alias = f"__{tag}_{letter}"
            if alias in out.columns:
                out = out.rename(columns={alias: PQR_DISPLAY.get(tag, {}).get(letter, alias)})

    # 2軍投手: SはK列由来の別名列
    if "__FARMP_S_K" in out.columns:
        out = out.rename(columns={"__FARMP_S_K": "S"})


    # formatting
    def fmt_val(v, col):
        # 重複列名などで v が配列/Series になるケースがあるため、先にスカラー化
        if isinstance(v, (pd.Series, list, tuple, np.ndarray)):
            try:
                arr = np.asarray(v)
                v = arr.ravel()[0] if arr.size else np.nan
            except Exception:
                v = np.nan
        if col in ["選手名","投打"]:
            return "" if pd.isna(v) else str(v)
        if col == "年齢":
            return "" if pd.isna(v) else str(int(float(v)))
        if col == "背番号":
            return "" if pd.isna(v) else str(v)
        if col == "回数":
            return fmt_ip(ip_to_float(v))
        if col == "防御率":
            return "" if pd.isna(v) else f"{float(v):.2f}"
        if col in ["K%","BB%","K-BB%","HR%"]:
            if pd.isna(v):
                return "-"
            # Excel 側で "12.3%" のような文字列になっているケースを許容
            if isinstance(v, str):
                s = v.strip().replace("％", "%")
                if s.endswith("%"):
                    s = s[:-1].strip()
                vv = pd.to_numeric(s, errors="coerce")
            else:
                vv = pd.to_numeric(v, errors="coerce")
            if pd.isna(vv):
                return "-"
            f = float(vv)
            # 0.123 のように割合(0-1)で入っている場合は % 表示に補正
            if 0 <= f <= 1:
                f *= 100
            return f"{f:.1f}%"
        if pd.isna(v):
            return "-"
        try:
            return str(int(round(float(v))))
        except Exception:
            return "-"

        # Preserve "-" cells from Excel (show "-" instead of blank)
    for c in out.columns:
        if c in d.columns:
            raw_s = d[c].astype(str)
            out[c] = [
                "-" if str(r).strip() in ["-", "－"] else fmt_val(x, c)
                for x, r in zip(out[c].values, raw_s.values)
            ]
        else:
            out[c] = out[c].apply(lambda x, cc=c: fmt_val(x, cc))

    # sort by innings desc then jersey
    out["__ip_sort"] = d["__ip_float"].values
    out["__jkey"] = d["__jkey"].values
    out = out.sort_values(["__ip_sort","__jkey"], ascending=[False, True]).drop(columns=["__ip_sort","__jkey"])
    return out

def pitcher_apps_for_level(merged_p):
    if merged_p is None or merged_p.empty:
        return None
    g_col = "試合" if "試合" in merged_p.columns else ("登板" if "登板" in merged_p.columns else None)
    if not g_col:
        return None
    p = merged_p[["選手ID", g_col]].copy().rename(columns={g_col:"登板"})
    p["登板"] = pd.to_numeric(p["登板"], errors="coerce").fillna(0)
    return p

def batter_table(df, pitch_apps_df=None, exclude_pitchers=True):
    d = df.copy()
    if "打席" in d.columns and "三振" in d.columns:
        d["K%"] = pct(d["三振"], d["打席"])
    else:
        d["K%"] = np.nan
    # BB% は（Excel側のBI列=BB%）を前提。
    # Excelの%書式は 0.075 (=7.5%) のように 0-1 の小数で読み込まれることが多いので、
    # 0-1 帯なら 0-100(%) に正規化してから表示する。
    if "BB%" in d.columns:
        bb = pd.to_numeric(d["BB%"], errors="coerce")
        if bb.notna().any() and float(bb.dropna().median()) <= 1.0:
            bb = bb * 100.0
        d["BB%"] = bb
    else:
        if "打席" in d.columns and "四球" in d.columns:
            d["BB%"] = pct(d["四球"], d["打席"])
        else:
            d["BB%"] = np.nan

    # exclude pitchers rule
    if pitch_apps_df is not None and "試合" in d.columns:
        d = d.merge(pitch_apps_df, on="選手ID", how="left")
        d["登板"] = d["登板"].fillna(0)
        if exclude_pitchers:
            bg = pd.to_numeric(d["試合"], errors="coerce").fillna(0)
            keep = (d["登板"] <= 0) | ((d["登板"] * 1.2) < bg)
            d = d[keep].copy()
    else:
        d["登板"] = 0


    # hide players with no batting appearances (試合=0 and 打席=0)
    gnum = pd.to_numeric(d["試合"], errors="coerce").fillna(0) if "試合" in d.columns else pd.Series([0]*len(d), index=d.index)
    panum = pd.to_numeric(d["打席"], errors="coerce").fillna(0) if "打席" in d.columns else pd.Series([0]*len(d), index=d.index)
    d = d[(gnum > 0) | (panum > 0)].copy()

    cols = ["背番号_use","選手名","年齢","投打"]
    for c in ["試合","打席","打数","得点","安打","二塁打","三塁打","本塁打","三振","四球","死球","盗塁","盗塁死","打率","出塁率","長打率","OPS","wOBA"]:
        if c in d.columns:
            cols.append(c)
    cols += ["K%","BB%"]

    out = d[cols].copy().rename(columns={"背番号_use":"背番号"})

    def fmt_val(v, col):
        if col in ["選手名","投打"]:
            return "" if pd.isna(v) else str(v)
        if col == "年齢":
            return "" if pd.isna(v) else str(int(float(v)))
        if col == "背番号":
            return "" if pd.isna(v) else str(v)
        if col in ["打率","出塁率","長打率","OPS","wOBA"]:
            return "" if pd.isna(v) else fmt_three_no_leading_zero(v)
        if col in ["K%","BB%"]:
            return "-" if pd.isna(v) else f"{float(v):.1f}%"
        if pd.isna(v):
            return "-"
        try:
            return str(int(round(float(v))))
        except Exception:
            return "-"

        # Preserve "-" cells from Excel (show "-" instead of blank)
    for c in out.columns:
        if c in d.columns:
            raw_s = d[c].astype(str)
            out[c] = [
                "-" if str(r).strip() in ["-", "－"] else fmt_val(x, c)
                for x, r in zip(out[c].values, raw_s.values)
            ]
        else:
            out[c] = out[c].apply(lambda x, cc=c: fmt_val(x, cc))

    # sort by PA desc then jersey
    if "打席" in d.columns:
        out["__pa_sort"] = pd.to_numeric(d["打席"], errors="coerce").fillna(0).values
    elif "試合" in d.columns:
        out["__pa_sort"] = pd.to_numeric(d["試合"], errors="coerce").fillna(0).values
    else:
        out["__pa_sort"] = 0
    out["__jkey"] = d["__jkey"].values
    out = out.sort_values(["__pa_sort","__jkey"], ascending=[False, True]).drop(columns=["__pa_sort","__jkey"])
    return out

IFRAME_CSS = """
<style>
  :root{ color-scheme: light; }
  body{ margin:0; padding:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Kaku Gothic ProN", "Meiryo", sans-serif; }
  .tbl-wrap{
    border: 1px solid rgba(49,51,63,0.14);
    border-radius: 16px;
    overflow: hidden;
    background: #fff;
    box-shadow: 0 12px 30px rgba(0,0,0,0.14);
  }
  .tbl-scroll{ overflow-x: auto; }
  table.npb{
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
    font-size: 12.5px;
  }
  table.npb thead th{
    position: sticky;
    top: 0;
    z-index: 30;
    background: #f2f3f5;
    box-shadow: 0 2px 0 rgba(49,51,63,0.22);
    font-weight: 900;
    text-align: center;
    padding: 8px 8px;
    border-bottom: 1px solid rgba(49,51,63,0.18);
    white-space: nowrap;
  }
  table.npb tbody td{
    padding: 6px 6px;
    border-bottom: 1px solid rgba(49,51,63,0.08);
    text-align: center;
    white-space: nowrap;
    background: #fff;
  }
  table.npb tbody tr:nth-child(2n) td{ background: rgba(49,51,63,0.02); }
  table.npb tbody tr:hover td{ background: rgba(0,119,255,0.06); }
  table.npb thead th, table.npb tbody td{ border-right: 1px solid rgba(49,51,63,0.12); }
  table.npb thead th:last-child, table.npb tbody td:last-child{ border-right: none; }
  th.sep, td.sep{ border-right: 4px double rgba(49,51,63,0.35) !important; }
  table.npb thead th.sortable{ cursor:pointer; user-select:none; }
  table.npb thead th.sortable:hover{ background: rgba(0,119,255,0.10); }
  td.left{ text-align:left !important; }
  th.num, td.num{ min-width: 54px; }
</style>
"""


def render_table(df: pd.DataFrame, title: str = "", table_id: str = "tbl",
                sticky_n: int = 4, sep_after: str | None = None,
                left_align_cols=None,  # e.g. {"回数"}
                left_cols=None, num_from_col=None, sep_after_col=None):
    """
    Render a DataFrame as an HTML table with:
      - sticky header row
      - sticky first `sticky_n` columns (freeze pane)
      - sortable columns (click header)
      - optional double separator line after `sep_after` columns
      - per-column left align (e.g., 回数)
    """
    # --- backward compatible arguments ---
    if sep_after is None and sep_after_col is not None:
        sep_after = sep_after_col
    if left_cols is not None:
        # left_cols can be list/set of column names to keep sticky; we translate to sticky_n
        try:
            idxs = [df.columns.get_loc(c) for c in left_cols if c in df.columns]
            if idxs:
                sticky_n = max(sticky_n, max(idxs) + 1)
        except Exception:
            pass
    # num_from_col is kept for compatibility; formatting is handled upstream

    left_align_cols = set(left_align_cols or [])

    cols = list(df.columns)
    n = len(cols)

    # sep_after can be an int (1-based) or a column name
    sep_after_idx = None
    if sep_after is not None:
        if isinstance(sep_after, int):
            sep_after_idx = sep_after
        else:
            try:
                sep_after_idx = cols.index(str(sep_after)) + 1
            except ValueError:
                sep_after_idx = None

    # Column widths (px) — keep compact to fit more columns
    # 1: 背番号, 2: 選手名, 3: 年齢, 4: 投打  → fixed
    base_widths = [72, 160, 64, 72]
    rest_w = 72
    widths = base_widths[:]
    if n > 4:
        widths += [rest_w] * (n - 4)

    # Build <colgroup>
    col_html = ['<colgroup>']
    for w in widths:
        col_html.append(f'<col style="width:{w}px;min-width:{w}px;">')
    col_html.append('</colgroup>')
    colgroup = ''.join(col_html)

    # Header
    html = []
    if title:
        html.append(f'<h3 style="margin:8px 0 12px 0;">{title}</h3>')
    html.append('<div class="npb-table-wrap">')
    html.append(f'<table class="npb-table">{colgroup}<thead><tr>')

    for i, c in enumerate(cols):
        cls = []
        if i < sticky_n:
            cls += ["sticky", f"col{i+1}"]
        if sep_after_idx and i == sep_after_idx - 1:
            cls.append("sep")
        cls_attr = f' class="{" ".join(cls)}"' if cls else ""
        html.append(f'<th{cls_attr} data-col="{i}">{c}<span class="sort-ind"> ⇅</span></th>')
    html.append('</tr></thead><tbody>')

    # Body
    for _, row in df.iterrows():
        html.append('<tr>')
        for i, c in enumerate(cols):
            v = row[c]
            s = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)

            cls = []
            if i < sticky_n:
                cls += ["sticky", f"col{i+1}"]
            if sep_after_idx and i == sep_after_idx - 1:
                cls.append("sep")
            if c in left_align_cols:
                cls.append("left")

            cls_attr = f' class="{" ".join(cls)}"' if cls else ""
            html.append(f'<td{cls_attr}>{s}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')

    # Sorting JS (client-side)
    html.append(r"""
<script>
(function(){
  const table = document.querySelector('.npb-table');
  if(!table) return;
  const tbody = table.querySelector('tbody');
  const ths = table.querySelectorAll('thead th');

  function parseVal(txt){
    if(txt === null) return NaN;
    let t = (''+txt).trim();
    if(t === '' || t === '-' || t === '—') return NaN;

    // innings like "146 2/3"
    const m = t.match(/^(\d+)\s+(\d)\/3$/);
    if(m){
      return parseFloat(m[1]) + (parseFloat(m[2]) / 3.0);
    }
    // percent like "21.1%"
    if(t.endsWith('%')){
      const p = parseFloat(t.replace('%',''));
      return isNaN(p) ? NaN : (p/100.0);
    }
    // leading dot like ".292"
    if(t.startsWith('.')) t = '0' + t;

    // remove commas
    t = t.replace(/,/g,'');
    const x = parseFloat(t);
    return isNaN(x) ? NaN : x;
  }

  function sortBy(col, asc){
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a,b)=>{
      const ta = a.children[col]?.innerText ?? '';
      const tb = b.children[col]?.innerText ?? '';
      const va = parseVal(ta);
      const vb = parseVal(tb);

      if(!isNaN(va) || !isNaN(vb)){
        if(isNaN(va) && isNaN(vb)) return 0;
        if(isNaN(va)) return 1;
        if(isNaN(vb)) return -1;
        return asc ? (va - vb) : (vb - va);
      }
      // fallback string compare (Japanese)
      return asc ? ta.localeCompare(tb, 'ja') : tb.localeCompare(ta, 'ja');
    });
    rows.forEach(r=>tbody.appendChild(r));
  }

  ths.forEach(th=>{
    th.addEventListener('click', ()=>{
      const col = parseInt(th.dataset.col, 10);
      const asc = th.dataset.asc !== '1';
      ths.forEach(x=>x.dataset.asc='');
      th.dataset.asc = asc ? '1' : '0';
      sortBy(col, asc);
    });
  });
})();
</script>
""")
    return "\n".join(html)

def unplayed_grid(roster, used_ids):
    r = roster[~roster["選手ID"].isin(used_ids)].copy()
    if r.empty:
        return []
    r2 = r[["背番号_use","選手名","年齢"]].copy().rename(columns={"背番号_use":"背番号"})
    items = []
    for _, row in r2.iterrows():
        items.append((str(row["背番号"]), str(row["選手名"]), str(int(row["年齢"])) if pd.notna(row["年齢"]) else ""))
    return items

# ----------------------------
# Streamlit UI

st.set_page_config(page_title="NPB 成績ダッシュボード", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
# ----------------------------

# Excel列(P/Q/R)の表示名を保持（シート読み込み時に埋める）
PQR_DISPLAY = {"TOP": {"Q": "S", "P": "H", "R": "HP"}, "FARM": {"Q": "S", "P": "H", "R": "HP"}, "TOPP": {"Q": "S", "P": "H", "R": "HP"}, "FARMP": {"Q": "S", "P": "H", "R": "HP"}}

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



def _parquet_cache_dir(excel_path: str) -> Path:
    p = Path(excel_path)
    try:
        sig = str(p.resolve()) + str(p.stat().st_mtime)
    except Exception:
        sig = str(p)
    h = hashlib.md5(sig.encode("utf-8")).hexdigest()[:12]
    d = Path(".cache_parquet") / h
    d.mkdir(parents=True, exist_ok=True)
    return d

def read_sheet_cached(excel_path: str, sheet_name: str) -> pd.DataFrame:
    """Read a sheet with Parquet cache (fast after first run)."""
    cache_dir = _parquet_cache_dir(excel_path)
    pq = cache_dir / f"{sheet_name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
    try:
        df.to_parquet(pq, index=False)
    except Exception:
        pass
    return df


def read_sheet_any(excel_path: str, sheet_names: List[str]) -> pd.DataFrame:
    """候補シート名のうち、読み込み可能な最初のものを返す。

    Excelのシート名が環境/バージョンで微妙に違うケースに備える。
    """
    last_err = None
    for name in sheet_names:
        try:
            return read_sheet_cached(excel_path, name)
        except Exception as e:
            last_err = e
            continue
    # ここまで来たら全部ダメ
    if last_err is not None:
        raise last_err
    raise ValueError("No sheet names provided")


st.markdown("""
<style>
.block-container{ padding-top: 1.2rem; padding-bottom: 3rem; }
.smallnote{ color: rgba(49,51,63,0.55); font-size: 12px; }
.card{
  border: 1px solid rgba(49,51,63,0.10);
  border-radius: 18px;
  padding: 14px 16px;
  background: #fff;
  box-shadow: 0 8px 22px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# Streamlitはウィジェット操作のたびに再実行されるため、
# サイドバーをフォーム化して「表示」クリック時のみ反映させ、体感のカクつきを抑えます。
if "_cfg" not in st.session_state:
    st.session_state["_cfg"] = {
        "page": "成績",
        "excel_path": "NPBデータ.xlsx",
        "team": "ソフトバンク",
        "year": 2025,
        "level": "1軍",
        "mode": "投手",
        "exclude_pitchers": True,
    }

cfg0 = st.session_state["_cfg"].copy()

# ------------------------------------------------------------
# Apply pending header changes BEFORE widgets are created.
# This avoids Streamlit warning:
# "The widget with key 'year' was created with a default value but also had its value set via the Session State API."
# ------------------------------------------------------------
if "_pending_hdr" in st.session_state:
    pend = st.session_state.pop("_pending_hdr")
    if isinstance(pend, dict):
        if "team" in pend:
            st.session_state["team"] = pend["team"]
            cfg0["team"] = pend["team"]
        if "year" in pend:
            st.session_state["year"] = pend["year"]
            cfg0["year"] = pend["year"]
        st.session_state["_cfg"] = cfg0.copy()
# ----------------------------
# Sidebar controls (use session_state keys so header can also drive them)
# ----------------------------
st.sidebar.title("メニュー")

_defaults = {
    "page": cfg0["page"],
    "excel_path": cfg0["excel_path"],
    "team": cfg0["team"],
    "year": cfg0.get("year"),
    "level": cfg0["level"],
    "mode": cfg0["mode"],
    "exclude_pitchers": cfg0.get("exclude_pitchers", True),
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

page = st.sidebar.selectbox("ページ", ["成績", "未出場"], key="page")
excel_path = st.sidebar.text_input("Excelファイルパス", key="excel_path")

team_options = ["ソフトバンク","日本ハム","オリックス","楽天","ロッテ","西武","巨人","阪神","中日","広島","ヤクルト","DeNA","オイシックス","ハヤテ"]
team = st.sidebar.selectbox(
    "チーム",
    team_options,
    key="team",
)

# ---- year options & session_state init (avoid Streamlit warning) ----
years = filter_team_year_options(team)
if not years:
    years = [2025]

# Used by the header "年度・チームを変更" expander
team_to_years = {t: (filter_team_year_options(t) or [2025]) for t in team_options}

if ("year" not in st.session_state) or (st.session_state["year"] not in years):
    st.session_state["year"] = years[-1]

year = st.sidebar.selectbox("年度", years, key="year")
level = st.sidebar.radio("1軍 / 2軍", ["1軍", "2軍"], horizontal=True, key="level")
mode = st.sidebar.radio("投手 / 打者", ["投手", "打者"], horizontal=True, key="mode")
exclude_pitchers = st.session_state.get("exclude_pitchers", True)
if mode == "打者":
    opt = st.sidebar.radio("打者表の投手", ["投手を除く", "投手を含む"], index=0 if exclude_pitchers else 1, horizontal=True, key="exclude_pitchers_ui")
    exclude_pitchers = (opt == "投手を除く")
    st.session_state["exclude_pitchers"] = exclude_pitchers
st.session_state["_cfg"] = {
    "page": page,
    "excel_path": excel_path,
    "team": team,
    "year": year,
    "level": level,
    "mode": mode,
    "exclude_pitchers": exclude_pitchers,
}
cfg = st.session_state["_cfg"]

aff, top_p, farm_p, top_b, farm_b = load_excel(cfg["excel_path"])

st.markdown(f"# {cfg['year']}年　{cfg['team']}")

# ヘッダーから年度・チームを変更できる簡易UI（クリック→プルダウンの代替）
with st.expander("年度・チームを変更", expanded=False):
    team_hdr = st.selectbox(
        "チーム",
        team_options,
        index=team_options.index(cfg["team"]) if cfg["team"] in team_options else 0,
        key="team_hdr",
    )
    years_hdr = team_to_years.get(team_hdr, years)
    year_hdr = st.selectbox(
        "年度",
        years_hdr,
        index=years_hdr.index(cfg["year"]) if cfg["year"] in years_hdr else 0,
        key="year_hdr",
    )
    if st.button("適用", key="hdr_apply"):
        # Set a pending dict that will be applied BEFORE widgets are created on the next rerun.
        st.session_state["_pending_hdr"] = {"team": team_hdr, "year": year_hdr}
        st.session_state["_cfg"]["team"] = team_hdr
        st.session_state["_cfg"]["year"] = year_hdr
        st.rerun()

roster = roster_for(aff, cfg["year"], cfg["team"])
p_sheet = top_p if cfg["level"] == "1軍" else farm_p
b_sheet = top_b if cfg["level"] == "1軍" else farm_b

merged_p = merge_roster_stats(roster, p_sheet, cfg["year"], cfg["team"])
merged_b = merge_roster_stats(roster, b_sheet, cfg["year"], cfg["team"])

pitch_apps = pitcher_apps_for_level(merged_p)


if cfg["page"] == "成績":
    if cfg["mode"] == "投手":
        st.markdown("## 投手成績")
        out = pitcher_table(merged_p)
        st.markdown(render_table(out, left_cols={"選手名"}, num_from_col="登板", sep_after_col="投打", left_align_cols={"回数"}), unsafe_allow_html=True)
    else:
        st.markdown("## 打者成績")
        out = batter_table(merged_b, pitch_apps_df=pitch_apps, exclude_pitchers=exclude_pitchers)
        st.markdown(render_table(out, left_cols={"選手名"}, num_from_col="試合", sep_after_col="投打"), unsafe_allow_html=True)
# ----------------------------
# 未出場ページ（この軍で 投手=登板なし かつ 打者=試合なし の選手）
# ----------------------------
if cfg["page"] == "未出場":
    st.markdown("## 未出場選手")
    # appearances in selected level
    p_apps = pitcher_apps_for_level(merged_p)
    p_any = set()
    if p_apps is not None:
        p_any = set(p_apps.loc[p_apps["登板"] > 0, "選手ID"].tolist())

    b_any = set()
    if not merged_b.empty and "試合" in merged_b.columns:
        b_any = set(merged_b.loc[pd.to_numeric(merged_b["試合"], errors="coerce").fillna(0) > 0, "選手ID"].tolist())

    used_any = p_any.union(b_any)
    r = roster[~roster["選手ID"].isin(used_any)].copy()

    if r.empty:
        st.caption("未出場選手はいません。")
    else:
        r2 = r[["背番号_use","選手名","年齢","投打"]].copy().rename(columns={"背番号_use":"背番号"})
        # grid
        cols = st.columns(6)
        for i, row in r2.iterrows():
            with cols[i % 6]:
                jno = "" if pd.isna(row["背番号"]) else str(row["背番号"])
                name = "" if pd.isna(row["選手名"]) else str(row["選手名"])
                age = "" if pd.isna(row["年齢"]) else str(int(float(row["年齢"])))
                tb = "" if pd.isna(row["投打"]) else str(row["投打"])
                st.markdown(f"""
<div class="card" style="text-align:center; padding:10px 10px;">
  <div style="font-weight:900; font-size:16px;">{jno}</div>
  <div style="font-weight:800; margin-top:2px;">{name}</div>
  <div class="smallnote">年齢 {age} / {tb}</div>
</div>
""", unsafe_allow_html=True)

    st.stop()