import sqlite3

con = sqlite3.connect("npb.sqlite")
cur = con.cursor()

print("\n--- tables & views in this DB ---")
cur.execute("SELECT type, name FROM sqlite_master WHERE type IN ('table','view') ORDER BY type, name")
rows = cur.fetchall()

for r in rows:
    print(r)

con.close()
