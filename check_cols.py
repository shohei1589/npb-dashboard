import sqlite3

con = sqlite3.connect("data/npb.sqlite")
cur = con.cursor()

print("\n--- pitching_2_raw columns ---")
cur.execute("PRAGMA table_info(pitching_2_raw)")
for r in cur.fetchall():
    print(r)

con.close()
