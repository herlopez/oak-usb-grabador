import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
SCRIPT_NAME = os.path.basename(__file__)
DEVICE_NAME = os.getenv("DEVICE_NAME")
DATABASE_FILE = os.getenv("DATABASE_FILE")
TABLE_NAME = os.getenv("TABLE_NAME")

def create_connection(db_file):
    if db_file is None:
        print("Warning: Se intentó crear conexión con db_file=None. Retornando None.")
        return -1
    
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except TypeError: # En caso de que sqlite3.connect reciba un tipo inesperado
        print(f"Error type connecting to database '{db_file}': Expected a string, got {type(db_file).__name__}")
        return -1
    except sqlite3.Error as e:
        print(f"Error connecting to database '{db_file}': {e}")
        return -1
    return conn

# Check if the table exists
def table_exists(conn, table_name):
    if conn is None: # <--- GUARDA CRUCIAL
        print("Warning: No connection provided to check table existence. Returning False.")
        return -1  # No hacer nada si la conexión es None
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        return cursor.fetchone() is not None  # True if table exists, False otherwise
    except sqlite3.Error as e:
        print(f"Error checking if table '{table_name}' exists: {e}")
        return -1  # Error checking table existence

def create_table(conn, table_name):
    if conn is None: # <--- GUARDA CRUCIAL
        print("Warning: No connection provided to create_table. Returning without action.")
        return -1# No hacer nada si la conexión es None

    sql_create_event_log_table = f""" CREATE TABLE IF NOT EXISTS {table_name} (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        timestamp REAL NOT NULL,
                                        device_name TEXT,
                                        script_name TEXT,
                                        event_type TEXT NOT NULL,
                                        pct_left REAL,
                                        pct_center REAL,
                                        pct_right REAL,
                                        pct_out_roi REAL,
                                        avg_count REAL,
                                        max_count REAL,
                                        filename TEXT,
                                        message TEXT
                                    ); """
    try:
        c = conn.cursor()
        c.execute(sql_create_event_log_table)
        return 0  # Success
    except sqlite3.Error as e:
        print(f"Error creating table 'event_log': {e}")
        return -1  # Error

def insert_event(conn, table_name, **kwargs):
    if conn is None: # <--- GUARDA CRUCIAL
        print("Warning: No connection provided to create_table. Returning without action.")
        return -1 # Retornar None si la conexión es None

    columns = ', '.join(kwargs.keys())
    placeholders = ', '.join(['?'] * len(kwargs))
    values = tuple(kwargs.values())
    sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

    try:
        cur = conn.cursor()
        cur.execute(sql, values)
        conn.commit()
        return cur.lastrowid  # Success: return row id
    except sqlite3.Error as e:
        print(f"Error inserting event into SQLite: {e}")
        return -1  # Error: return -1 to indicate failure

if __name__ == '__main__':
    print(f"Probando sql_logger.py. Usando base de datos: {DATABASE_FILE}")
    if DATABASE_FILE is None:
        print("Error: DATABASE_FILE es None en __main__. Verifica .env o el valor por defecto.")
    else:
        conn_test = create_connection(DATABASE_FILE)
        if conn_test != -1:
            create_table(conn_test, TABLE_NAME)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            test_event_id = insert_event(conn_test, TABLE_NAME, (ts, DEVICE_NAME, SCRIPT_NAME, "TEST_EVENT", 123.45, "This is a test message."))
            if test_event_id != -1:
                print(f"Test event ID: {test_event_id}")
            
                # Leer y mostrar el evento insertado (opcional)
                cursor = conn_test.cursor()
                cursor.execute("SELECT * FROM event_log WHERE id = ?", (test_event_id,))
                row = cursor.fetchone()
                if row:
                    print(f"Event recovered: {row}")
                else:
                    print("Cannot recover test event.")                    
            else:
                print("Test event not inserted (ID is None).")

            conn_test.close()
            print("Test connection closed.")
        else:
            print("Failed to create test connection.")
