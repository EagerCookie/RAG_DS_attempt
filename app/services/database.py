"""
Database Manager - Handles SQLite operations for tracking processed files and settings
Updated to support multiple vector databases per pipeline
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
            
            # Pipelines table - store pipeline configurations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pipelines (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config TEXT NOT NULL,
                    vector_db_identifier TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Files table - track processed files with pipeline and vector DB association
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER,
                    pipeline_id TEXT NOT NULL,
                    vector_db_identifier TEXT NOT NULL,
                    chunks_count INTEGER,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (pipeline_id) REFERENCES pipelines(id) ON DELETE CASCADE,
                    UNIQUE(file_hash, pipeline_id)
                )
            ''')
            
            # Processing tasks table - track background jobs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    pipeline_id TEXT,
                    filename TEXT,
                    file_hash TEXT,
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
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_pipeline ON files(pipeline_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_vector_db ON files(vector_db_identifier)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pipelines_name ON pipelines(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pipelines_vector_db ON pipelines(vector_db_identifier)')
    
    # ==========================================
    # PIPELINE OPERATIONS
    # ==========================================
    
    def save_pipeline(self, pipeline_id: str, name: str, config: str, vector_db_identifier: str):
        """Save or update pipeline configuration with its vector DB identifier"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO pipelines (id, name, config, vector_db_identifier)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    config = excluded.config,
                    vector_db_identifier = excluded.vector_db_identifier,
                    updated_at = CURRENT_TIMESTAMP
            ''', (pipeline_id, name, config, vector_db_identifier))
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Dict]:
        """Get pipeline by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, name, config, vector_db_identifier, created_at, updated_at FROM pipelines WHERE id = ?',
                (pipeline_id,)
            )
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def list_pipelines(self) -> List[Dict]:
        """List all pipelines"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, vector_db_identifier, created_at, updated_at
                FROM pipelines
                ORDER BY updated_at DESC
            ''')
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete pipeline and all associated files"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM pipelines WHERE id = ?', (pipeline_id,))
            return cursor.rowcount > 0
    
    def get_pipeline_by_vector_db(self, vector_db_identifier: str) -> List[Dict]:
        """Get all pipelines using specific vector database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, config, vector_db_identifier, created_at, updated_at
                FROM pipelines
                WHERE vector_db_identifier = ?
                ORDER BY created_at DESC
            ''', (vector_db_identifier,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ==========================================
    # FILE OPERATIONS
    # ==========================================
    
    def file_exists_in_pipeline(self, file_hash: str, pipeline_id: str) -> Optional[Tuple[str, str]]:
        """
        Check if file with given hash exists in specific pipeline
        
        Returns:
            Tuple of (filename, processed_at) if exists, None otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT filename, processed_at FROM files WHERE file_hash = ? AND pipeline_id = ?',
                (file_hash, pipeline_id)
            )
            result = cursor.fetchone()
            return (result['filename'], result['processed_at']) if result else None
    
    def file_exists_in_vector_db(self, file_hash: str, vector_db_identifier: str) -> List[Dict]:
        """
        Get all occurrences of file in specific vector database (across all pipelines)
        
        Returns:
            List of dicts with file info
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT f.id, f.filename, f.file_hash, f.pipeline_id, p.name as pipeline_name,
                       f.processed_at, f.chunks_count
                FROM files f
                JOIN pipelines p ON f.pipeline_id = p.id
                WHERE f.file_hash = ? AND f.vector_db_identifier = ?
            ''', (file_hash, vector_db_identifier))
            return [dict(row) for row in cursor.fetchall()]
    
    def add_file(
        self,
        filename: str,
        file_hash: str,
        file_size: int,
        pipeline_id: str,
        vector_db_identifier: str,
        chunks_count: int,
        metadata: Optional[Dict] = None
    ):
        """Add processed file record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO files (filename, file_hash, file_size, pipeline_id, 
                                   vector_db_identifier, chunks_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                filename,
                file_hash,
                file_size,
                pipeline_id,
                vector_db_identifier,
                chunks_count,
                json.dumps(metadata) if metadata else None
            ))
    
    def get_files_by_pipeline(self, pipeline_id: str, limit: int = 100) -> List[Dict]:
        """Get all files processed by specific pipeline"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, filename, file_hash, file_size, pipeline_id, 
                       vector_db_identifier, chunks_count, processed_at, metadata
                FROM files
                WHERE pipeline_id = ?
                ORDER BY processed_at DESC
                LIMIT ?
            ''', (pipeline_id, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_files_by_vector_db(self, vector_db_identifier: str, limit: int = 100) -> List[Dict]:
        """Get all files in specific vector database (across all pipelines)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT f.id, f.filename, f.file_hash, f.file_size, 
                       f.pipeline_id, p.name as pipeline_name,
                       f.vector_db_identifier, f.chunks_count, 
                       f.processed_at, f.metadata
                FROM files f
                JOIN pipelines p ON f.pipeline_id = p.id
                WHERE f.vector_db_identifier = ?
                ORDER BY f.processed_at DESC
                LIMIT ?
            ''', (vector_db_identifier, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_processed_files(self, limit: int = 100) -> List[Dict]:
        """Get list of all processed files"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT f.id, f.filename, f.file_hash, f.file_size, 
                       f.pipeline_id, p.name as pipeline_name,
                       f.vector_db_identifier, f.chunks_count, 
                       f.processed_at, f.metadata
                FROM files f
                JOIN pipelines p ON f.pipeline_id = p.id
                ORDER BY f.processed_at DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_file_record(self, file_id: int) -> bool:
        """Delete file record by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM files WHERE id = ?', (file_id,))
            return cursor.rowcount > 0
    
    def get_file_duplicates(self, file_hash: str) -> List[Dict]:
        """Get all occurrences of a file across all pipelines"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT f.id, f.filename, f.file_hash, 
                       f.pipeline_id, p.name as pipeline_name,
                       f.vector_db_identifier, f.chunks_count, 
                       f.processed_at
                FROM files f
                JOIN pipelines p ON f.pipeline_id = p.id
                WHERE f.file_hash = ?
                ORDER BY f.processed_at DESC
            ''', (file_hash,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ==========================================
    # TASK OPERATIONS
    # ==========================================
    
    def create_task(
        self,
        task_id: str,
        pipeline_id: str,
        filename: str,
        file_hash: str,
        status: str = "pending"
    ):
        """Create new processing task"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO tasks (id, pipeline_id, filename, file_hash, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (task_id, pipeline_id, filename, file_hash, status))
    
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
                SELECT id, pipeline_id, filename, file_hash, status, progress, 
                       message, error, created_at, updated_at
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
                    SELECT id, pipeline_id, filename, file_hash, status, progress, 
                           message, error, created_at, updated_at
                    FROM tasks
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (status, limit))
            else:
                cursor.execute('''
                    SELECT id, pipeline_id, filename, file_hash, status, progress, 
                           message, error, created_at, updated_at
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
    # STATISTICS & ANALYTICS
    # ==========================================
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total files processed
            cursor.execute('SELECT COUNT(*) as count FROM files')
            stats['total_files'] = cursor.fetchone()['count']
            
            # Total unique files (by hash)
            cursor.execute('SELECT COUNT(DISTINCT file_hash) as count FROM files')
            stats['unique_files'] = cursor.fetchone()['count']
            
            # Total pipelines
            cursor.execute('SELECT COUNT(*) as count FROM pipelines')
            stats['total_pipelines'] = cursor.fetchone()['count']
            
            # Total vector databases
            cursor.execute('SELECT COUNT(DISTINCT vector_db_identifier) as count FROM pipelines')
            stats['total_vector_dbs'] = cursor.fetchone()['count']
            
            # Total chunks created
            cursor.execute('SELECT SUM(chunks_count) as total FROM files')
            result = cursor.fetchone()
            stats['total_chunks'] = result['total'] if result['total'] else 0
            
            # Files by pipeline
            cursor.execute('''
                SELECT p.name, COUNT(f.id) as count
                FROM pipelines p
                LEFT JOIN files f ON p.id = f.pipeline_id
                GROUP BY p.id, p.name
            ''')
            stats['files_by_pipeline'] = {
                row['name']: row['count'] 
                for row in cursor.fetchall()
            }
            
            # Files by vector database
            cursor.execute('''
                SELECT vector_db_identifier, COUNT(*) as count
                FROM files
                GROUP BY vector_db_identifier
            ''')
            stats['files_by_vector_db'] = {
                row['vector_db_identifier']: row['count'] 
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
    
    def get_vector_db_statistics(self, vector_db_identifier: str) -> Dict:
        """Get statistics for specific vector database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {'vector_db_identifier': vector_db_identifier}
            
            # Total files in this DB
            cursor.execute('''
                SELECT COUNT(*) as count FROM files WHERE vector_db_identifier = ?
            ''', (vector_db_identifier,))
            stats['total_files'] = cursor.fetchone()['count']
            
            # Unique files (by hash)
            cursor.execute('''
                SELECT COUNT(DISTINCT file_hash) as count 
                FROM files 
                WHERE vector_db_identifier = ?
            ''', (vector_db_identifier,))
            stats['unique_files'] = cursor.fetchone()['count']
            
            # Total chunks
            cursor.execute('''
                SELECT SUM(chunks_count) as total 
                FROM files 
                WHERE vector_db_identifier = ?
            ''', (vector_db_identifier,))
            result = cursor.fetchone()
            stats['total_chunks'] = result['total'] if result['total'] else 0
            
            # Pipelines using this DB
            cursor.execute('''
                SELECT id, name 
                FROM pipelines 
                WHERE vector_db_identifier = ?
            ''', (vector_db_identifier,))
            stats['pipelines'] = [dict(row) for row in cursor.fetchall()]
            
            return stats