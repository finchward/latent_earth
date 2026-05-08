import os
import sqlite3
import argparse

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "metadata.sqlite")

def reset_flag(flag_name):
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        # We use string formatting for column name as parameter binding is for values only
        conn.execute(f"UPDATE metadata SET {flag_name} = 0")
        conn.commit()
        print(f"✅ Successfully reset '{flag_name}' to 0 for all records.")
    except sqlite3.OperationalError as e:
        print(f"❌ Error resetting '{flag_name}': {e}")
        print("   Are you sure that column exists in the metadata table?")
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset indexing flags in the metadata database.")
    parser.add_argument(
        "--flag", 
        type=str, 
        default="indexed_hog_hybrid",
        help="The name of the flag column to reset (default: indexed_hog_hybrid)"
    )
    
    args = parser.parse_args()
    reset_flag(args.flag)
