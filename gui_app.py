import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import os
import threading
import hashlib
import sqlite3
import json
from dotenv import load_dotenv

# Langchain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# from langchain_qdrant import Qdrant # Uncomment when Qdrant is installed

# Load environment variables
load_dotenv()

DB_NAME = "rag_data.db"
CONFIG_FILE = "app_config.json"

class DatabaseManager:
    def __init__(self, db_name=DB_NAME):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            # Table for processed files
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Table for system settings (singleton row)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    embedding_model TEXT,
                    splitter_type TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def file_exists(self, file_hash):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT filename FROM files WHERE file_hash = ?', (file_hash,))
            return cursor.fetchone()

    def add_file(self, filename, file_hash):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO files (filename, file_hash) VALUES (?, ?)', (filename, file_hash))
            conn.commit()

    def get_settings(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT embedding_model, splitter_type FROM settings WHERE id = 1')
            return cursor.fetchone()

    def save_settings(self, embedding_model, splitter_type):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            # Use INSERT OR IGNORE to ensure we only ever have one row (id=1)
            cursor.execute('''
                INSERT OR IGNORE INTO settings (id, embedding_model, splitter_type) 
                VALUES (1, ?, ?)
            ''', (embedding_model, splitter_type))
            conn.commit()

def calculate_file_hash(filepath):
    """Calculates MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_local_config():
    """Loads UI settings from JSON file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_local_config(config):
    """Saves UI settings to JSON file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)


# ==========================================
# CONFIGURATION TEMPLATES
# ==========================================

# 1. LOADER OPTIONS
# Format: "Display Name": {"class": LoaderClass, "ext": "file_extension_filter"}
LOADER_OPTIONS = {
    "PyPDFLoader": {
        "class": PyPDFLoader,
        "ext": "*.pdf",
        "description": "Loads PDF files"
    },
    # TEMPLATE: Add new loaders here
    # "TextLoader": {
    #     "class": TextLoader,
    #     "ext": "*.txt",
    #     "description": "Loads text files"
    # },
}

# 2. SPLITTER OPTIONS
# Format: "Display Name": function_that_returns_splitter_instance
SPLITTER_OPTIONS = {
    "Recursive Character Splitter": lambda: RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True
    ),
    "Sentence Transformers Splitter": lambda: SentenceTransformersTokenTextSplitter(
        model_name="DeepVk/USER-bge-m3",
        chunk_size=256,
        chunk_overlap=50
    ),
    # TEMPLATE: Add new splitters here
}

# 3. EMBEDDING OPTIONS
# Format: "Display Name": {"model_name": "huggingface/model-name", "kwargs": {...}}
EMBEDDING_OPTIONS = {
    "DeepVk/USER-bge-m3": {
        "model_name": "DeepVk/USER-bge-m3",
        "model_kwargs": {'device': 'cpu'},
        "encode_kwargs": {'normalize_embeddings': True}
    },
    "cointegrated/rubert-tiny2": {
        "model_name": "cointegrated/rubert-tiny2",
        "model_kwargs": {'device': 'cpu'},
        "encode_kwargs": {'normalize_embeddings': True}
    },
    # TEMPLATE: Add new embedding models here
}

# 4. VECTOR DATABASE OPTIONS
# Format: "Display Name": function_that_takes_(documents, embeddings)_and_returns_db_or_ids
def save_to_chroma(documents, embeddings):
    return Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    ).add_documents(documents)

def save_to_qdrant(documents, embeddings):
    # TEMPLATE: Implement Qdrant logic here
    # Example:
    # return Qdrant.from_documents(
    #     documents,
    #     embeddings,
    #     url="http://localhost:6333",
    #     collection_name="my_documents"
    # )
    raise NotImplementedError("Qdrant logic needs to be implemented in the template.")

VECTOR_DB_OPTIONS = {
    "Chroma (Local)": save_to_chroma,
    "Qdrant (Template)": save_to_qdrant,
    # TEMPLATE: Add new databases here
}

# ==========================================
# GUI APPLICATION
# ==========================================

class DocumentProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Document Processor")
        self.root.geometry("600x750")

        self.db_manager = DatabaseManager()
        self.local_config = load_local_config()
        self.selected_file_path = None

        self._create_widgets()

    def _create_widgets(self):
        # --- File Selection Section ---
        file_frame = ttk.LabelFrame(self.root, text="1. Select File", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)

        # Load default loader from config or use first option
        default_loader = self.local_config.get("loader", list(LOADER_OPTIONS.keys())[0])
        self.loader_var = tk.StringVar(value=default_loader)
        
        ttk.Label(file_frame, text="Loader Type:").pack(anchor="w")
        self.loader_combo = ttk.Combobox(file_frame, textvariable=self.loader_var, values=list(LOADER_OPTIONS.keys()), state="readonly")
        self.loader_combo.pack(fill="x", pady=5)
        
        file_btn_frame = ttk.Frame(file_frame)
        file_btn_frame.pack(fill="x", pady=5)
        
        self.file_btn = ttk.Button(file_btn_frame, text="Browse File...", command=self.select_file)
        self.file_btn.pack(side="left")
        
        self.file_label = ttk.Label(file_btn_frame, text="No file selected", foreground="gray")
        self.file_label.pack(side="left", padx=10)

        # --- Splitter Section ---
        splitter_frame = ttk.LabelFrame(self.root, text="2. Select Text Splitter", padding=10)
        splitter_frame.pack(fill="x", padx=10, pady=5)

        default_splitter = self.local_config.get("splitter", list(SPLITTER_OPTIONS.keys())[0])
        self.splitter_var = tk.StringVar(value=default_splitter)
        self.splitter_combo = ttk.Combobox(splitter_frame, textvariable=self.splitter_var, values=list(SPLITTER_OPTIONS.keys()), state="readonly")
        self.splitter_combo.pack(fill="x", pady=5)

        # --- Embedding Section ---
        embed_frame = ttk.LabelFrame(self.root, text="3. Select Embedding Model", padding=10)
        embed_frame.pack(fill="x", padx=10, pady=5)

        default_embed = self.local_config.get("embedding", list(EMBEDDING_OPTIONS.keys())[0])
        self.embed_var = tk.StringVar(value=default_embed)
        self.embed_combo = ttk.Combobox(embed_frame, textvariable=self.embed_var, values=list(EMBEDDING_OPTIONS.keys()), state="readonly")
        self.embed_combo.pack(fill="x", pady=5)

        # --- Database Section ---
        db_frame = ttk.LabelFrame(self.root, text="4. Select Vector Database", padding=10)
        db_frame.pack(fill="x", padx=10, pady=5)

        default_db = self.local_config.get("database", list(VECTOR_DB_OPTIONS.keys())[0])
        self.db_var = tk.StringVar(value=default_db)
        self.db_combo = ttk.Combobox(db_frame, textvariable=self.db_var, values=list(VECTOR_DB_OPTIONS.keys()), state="readonly")
        self.db_combo.pack(fill="x", pady=5)

        # --- Process Button ---
        self.process_btn = ttk.Button(self.root, text="PROCESS DOCUMENT", command=self.start_processing)
        self.process_btn.pack(fill="x", padx=20, pady=15)

        # --- Log Area ---
        log_frame = ttk.LabelFrame(self.root, text="Logs", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.log_area = scrolledtext.ScrolledText(log_frame, height=10, state='disabled')
        self.log_area.pack(fill="both", expand=True)

    def log(self, message):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    def select_file(self):
        loader_name = self.loader_var.get()
        loader_info = LOADER_OPTIONS.get(loader_name)
        
        if not loader_info:
            self.log("Error: Invalid loader selected.")
            return

        filetypes = [(f"{loader_name} files", loader_info["ext"]), ("All files", "*.*")]
        
        filename = filedialog.askopenfilename(title="Select Document", filetypes=filetypes)
        
        if filename:
            self.selected_file_path = filename
            self.file_label.config(text=os.path.basename(filename), foreground="black")
            self.log(f"Selected file: {filename}")

    def start_processing(self):
        if not self.selected_file_path:
            self.log("Error: Please select a file first.")
            return
        
        # Disable button during processing
        self.process_btn.config(state="disabled")
        
        # Run in a separate thread to keep GUI responsive
        thread = threading.Thread(target=self.process_document)
        thread.start()

    def process_document(self):
        try:
            self.log("--- Starting Processing ---")
            
            # 0. Validation & Persistence Logic
            filepath = self.selected_file_path
            filename = os.path.basename(filepath)
            
            # Calculate Hash
            self.log("Calculating file hash...")
            file_hash = calculate_file_hash(filepath)
            self.log(f"File Hash: {file_hash}")

            # Check Duplicate
            existing_file = self.db_manager.file_exists(file_hash)
            if existing_file:
                msg = f"File '{existing_file[0]}' with this hash already exists in the database!"
                self.log(f"WARNING: {msg}")
                messagebox.showwarning("Duplicate File", msg)
                return # Stop processing

            # Check Settings Consistency
            current_embed = self.embed_var.get()
            current_splitter = self.splitter_var.get()
            
            saved_settings = self.db_manager.get_settings()
            
            if saved_settings:
                saved_embed, saved_splitter = saved_settings
                if saved_embed != current_embed or saved_splitter != current_splitter:
                    msg = (f"Settings Mismatch!\n\n"
                           f"Saved: Embed='{saved_embed}', Splitter='{saved_splitter}'\n"
                           f"Current: Embed='{current_embed}', Splitter='{current_splitter}'\n\n"
                           f"Using different settings for the same database is not recommended.\n"
                           f"Do you want to continue?")
                    if not messagebox.askyesno("Settings Mismatch", msg):
                        self.log("Processing cancelled by user due to settings mismatch.")
                        return
            else:
                # First time run, save these settings
                self.db_manager.save_settings(current_embed, current_splitter)

            # 1. Load
            loader_name = self.loader_var.get()
            loader_class = LOADER_OPTIONS[loader_name]["class"]
            self.log(f"Loading document using {loader_name}...")
            loader = loader_class(self.selected_file_path)
            docs = loader.load()
            self.log(f"Loaded {len(docs)} pages/documents.")

            # 2. Split
            splitter_name = self.splitter_var.get()
            self.log(f"Splitting text using {splitter_name}...")
            text_splitter = SPLITTER_OPTIONS[splitter_name]()
            all_splits = text_splitter.split_documents(docs)
            self.log(f"Created {len(all_splits)} chunks.")

            # 3. Embeddings
            embed_name = self.embed_var.get()
            self.log(f"Initializing embeddings: {embed_name}...")
            embed_config = EMBEDDING_OPTIONS[embed_name]
            embeddings = HuggingFaceEmbeddings(
                model_name=embed_config["model_name"],
                model_kwargs=embed_config["model_kwargs"],
                encode_kwargs=embed_config["encode_kwargs"],
                cache_folder="./transformers_models"
            )

            # 4. Vector DB
            db_name = self.db_var.get()
            self.log(f"Saving to Vector DB: {db_name}...")
            save_func = VECTOR_DB_OPTIONS[db_name]
            
            # Execute save
            save_func(all_splits, embeddings)
            
            # 5. Post-Processing Success
            self.db_manager.add_file(filename, file_hash)
            
            # Save local config for next time
            new_config = {
                "loader": loader_name,
                "splitter": splitter_name,
                "embedding": embed_name,
                "database": db_name
            }
            save_local_config(new_config)

            # Verification log
            test_vec = embeddings.embed_query("test")
            self.log(f"Success! Documents added to {db_name}.")
            self.log(f"Embedding dimension check: {len(test_vec)}")
            self.log(f"File recorded in database.")

        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.process_btn.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentProcessorApp(root)
    root.mainloop()
