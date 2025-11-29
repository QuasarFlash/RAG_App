import sqlite3
import os
from datetime import datetime
import uuid

class ChatDatabase:
    def __init__(self, db_path="./chat_history.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Create tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Table for Sessions (The "folders" of chats)
            c.execute('''CREATE TABLE IF NOT EXISTS sessions
                         (id TEXT PRIMARY KEY, title TEXT, timestamp DATETIME)''')
            
            # Table for Messages (The actual content)
            c.execute('''CREATE TABLE IF NOT EXISTS messages
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                          session_id TEXT, 
                          role TEXT, 
                          content TEXT, 
                          image_path TEXT, 
                          timestamp DATETIME,
                          FOREIGN KEY(session_id) REFERENCES sessions(id))''')
            conn.commit()

    def create_session(self, title=None):
        session_id = str(uuid.uuid4())
        if not title:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO sessions (id, title, timestamp) VALUES (?, ?, ?)",
                         (session_id, title, datetime.now()))
        return session_id

    def add_message(self, session_id, role, content, image_path=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''INSERT INTO messages (session_id, role, content, image_path, timestamp) 
                            VALUES (?, ?, ?, ?, ?)''', 
                            (session_id, role, content, image_path, datetime.now()))

    def get_all_sessions(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Order by newest first
            cursor = conn.execute("SELECT * FROM sessions ORDER BY timestamp DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_messages(self, session_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
            return [dict(row) for row in cursor.fetchall()]

    def delete_session(self, session_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    
    def search_sessions(self, query):
        """
        Search for sessions where the title OR any message content matches the query.
        Returns a list of unique sessions.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # We use DISTINCT because one chat might have 5 messages matching the query,
            # but we only want to list the session once.
            sql = '''
                SELECT DISTINCT s.id, s.title, s.timestamp
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE s.title LIKE ? OR m.content LIKE ?
                ORDER BY s.timestamp DESC
            '''
            wildcard_query = f"%{query}%"
            cursor = conn.execute(sql, (wildcard_query, wildcard_query))
            return [dict(row) for row in cursor.fetchall()]