from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = PROJECT_ROOT / "NPBデータ.xlsx"

SHEET = "選手所属"  # ここだけ確認したい

df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET)

print("=== sheet:", SHEET, "===")
print("columns:")
for c in df.columns:
    print(repr(str(c)))
print("\nhead:")
print(df.head(3))
