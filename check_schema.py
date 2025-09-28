import sqlite3
import os

basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, "wattcast.db")

print("Checking DB file at:", db_path)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in DB:", tables)

if ('user',) in tables:
    cursor.execute("PRAGMA table_info(user);")
    columns = cursor.fetchall()
    print("Columns in 'user' table:")
    for col in columns:
        print(f"  {col}")
else:
    print("Table 'user' does not exist.")

conn.close()

