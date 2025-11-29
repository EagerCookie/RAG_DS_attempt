"""
Database Manager - Handles SQLite operations for tracking processed files and settings
"""
import sqlite3
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from contextlib import contextmanager
import json


class DatabaseManager:
    """Manages SQLite database for file tracking and settings"""
    
    def __init__(self, db_path: str = "rag_data.db"):
        self.db_path = db_path
        self.init_db()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_db(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Files table - track processed files
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    file_size INTEGER,
                    pipeline_id TEXT,
                    chunks_count INTEGER,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Pipelines table - store pipeline configurations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pipelines (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Processing tasks table - track background jobs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    pipeline_id TEXT,
                    filename TEXT,
                    status TEXT,
                    progress REAL,
                    message TEXT,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pipeline_id) REFERENCES pipelines(id)
                )
            ''')
            
            # Settings table - store global settings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pipelines_name ON pipelines(name)')
    
    # ==========================================
    # FILE OPERATIONS
    # ==========================================
    
    def file_exists(self, file_hash: str) -> Optional[Tuple[str, str]]:
        """
        Check if file with given hash exists
        
        Returns:
            Tuple of (filename, processed_at) if exists, None otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT filename, processed_at FROM files WHERE file_hash = ?',
                (file_hash,)
            )
            result = cursor.fetchone()
            return (result['filename'], result['processed_at']) if result else None
    
    def add_file(
        self,
        filename: str,
        file_hash: str,
        file_size: int,
        pipeline_id: str,
        chunks_count: int,
        metadata: Optional[Dict] = None
    ):
        """Add processed file record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO files (filename, file_hash, file_size, pipeline_id, chunks_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                filename,
                file_hash,
                file_size,
                pipeline_id,
                chunks_count,
                json.dumps(metadata) if metadata else None
            ))
    
    def get_processed_files(self, limit: int = 100) -> List[Dict]:
        """Get list of processed files"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, filename, file_hash, file_size, pipeline_id, 
                       chunks_count, processed_at, metadata
                FROM files
                ORDER BY processed_at DESC
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_file_record(self, file_hash: str) -> bool:
        """Delete file record (doesn't delete from vector DB)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM files WHERE file_hash = ?', (file_hash,))
            return cursor.rowcount > 0
    
    # ==========================================
    # PIPELINE OPERATIONS
    # ==========================================
    
    def save_pipeline(self, pipeline_id: str, name: str, config: str):
        """Save or update pipeline configuration"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO pipelines (id, name, config)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    config = excluded.config,
                    updated_at = CURRENT_TIMESTAMP
            ''', (pipeline_id, name, config))
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Dict]:
        """Get pipeline by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, name, config, created_at, updated_at FROM pipelines WHERE id = ?',
                (pipeline_id,)
            )
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def list_pipelines(self) -> List[Dict]:
        """List all pipelines"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, created_at, updated_at
                FROM pipelines
                ORDER BY updated_at DESC
            ''')
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete pipeline"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM pipelines WHERE id = ?', (pipeline_id,))
            return cursor.rowcount > 0
    
    # ==========================================
    # TASK OPERATIONS
    # ==========================================
    
    def create_task(
        self,
        task_id: str,
        pipeline_id: str,
        filename: str,
        status: str = "pending"
    ):
        """Create new processing task"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO tasks (id, pipeline_id, filename, status)
                VALUES (?, ?, ?, ?)
            ''', (task_id, pipeline_id, filename, status))
    
    def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Update task status"""
        updates = []
        params = []
        
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if progress is not None:
            updates.append("progress = ?")
            params.append(progress)
        if message is not None:
            updates.append("message = ?")
            params.append(message)
        if error is not None:
            updates.append("error = ?")
            params.append(error)
        
        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(task_id)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    UPDATE tasks
                    SET {", ".join(updates)}
                    WHERE id = ?
                ''', params)
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, pipeline_id, filename, status, progress, message, error,
                       created_at, updated_at
                FROM tasks
                WHERE id = ?
            ''', (task_id,))
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def list_tasks(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """List tasks, optionally filtered by status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute('''
                    SELECT id, pipeline_id, filename, status, progress, message, error,
                           created_at, updated_at
                    FROM tasks
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (status, limit))
            else:
                cursor.execute('''
                    SELECT id, pipeline_id, filename, status, progress, message, error,
                           created_at, updated_at
                    FROM tasks
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # ==========================================
    # SETTINGS OPERATIONS
    # ==========================================
    
    def get_setting(self, key: str) -> Optional[str]:
        """Get setting value"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))
            result = cursor.fetchone()
            return result['value'] if result else None
    
    def set_setting(self, key: str, value: str):
        """Set setting value"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO settings (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
            ''', (key, value))
    
    def get_all_settings(self) -> Dict[str, str]:
        """Get all settings"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM settings')
            return {row['key']: row['value'] for row in cursor.fetchall()}
    
    # ==========================================
    # STATISTICS
    # ==========================================
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total files processed
            cursor.execute('SELECT COUNT(*) as count FROM files')
            stats['total_files'] = cursor.fetchone()['count']
            
            # Total chunks created
            cursor.execute('SELECT SUM(chunks_count) as total FROM files')
            result = cursor.fetchone()
            stats['total_chunks'] = result['total'] if result['total'] else 0
            
            # Files by pipeline
            cursor.execute('''
                SELECT pipeline_id, COUNT(*) as count
                FROM files
                GROUP BY pipeline_id
            ''')
            stats['files_by_pipeline'] = {
                row['pipeline_id']: row['count'] 
                for row in cursor.fetchall()
            }
            
            # Recent processing activity (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) as count
                FROM files
                WHERE processed_at > datetime('now', '-1 day')
            ''')
            stats['recent_files'] = cursor.fetchone()['count']
            
            return stats