import os
import sys

os.chdir(r'C:\Users\zheng\Desktop\toBen\30.007_updated')
sys.path.insert(0, os.path.join(os.getcwd(), '30.007'))

from Database.db import get_connection, ensure_patient

conn = get_connection()
print('patients schema:')
print(conn.execute('PRAGMA table_info(patients)').fetchall())
print('sessions schema:')
print(conn.execute('PRAGMA table_info(sessions)').fetchall())
print('foreign keys:')
print(conn.execute('PRAGMA foreign_key_list(sessions)').fetchall())
print('patient rows before:', conn.execute('SELECT patient_id FROM patients').fetchall())
ensure_patient('demo_patient')
print('patient rows after:', conn.execute('SELECT patient_id FROM patients').fetchall())
try:
    conn.execute('INSERT INTO sessions(session_id, patient_id, session_date, started_at, ended_at, administered_by, photo_path) VALUES (?, ?, ?, ?, ?, ?, ?)', ('x1', 'demo_patient', '2026-01-01', '2026-01-01', '2026-01-01', 'tester', ''))
    conn.commit()
    print('insert succeeded')
except Exception as e:
    print('insert failed', repr(e))
finally:
    conn.close()
