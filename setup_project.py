#!/usr/bin/env python3
"""
Скрипт для автоматической настройки структуры проекта RAG Pipeline API
Запуск: python setup_project.py
"""
import os
import sys
from pathlib import Path


def create_directory_structure():
    """Создает структуру директорий"""
    dirs = [
        "app",
        "app/models",
        "app/services",
        "app/api",
        "app/api/endpoints",
        "app/core",
        "app/utils",
        "data",
        "data/chroma_langchain_db",
        "data/transformers_models",
        "tests",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Создана директория: {dir_path}")


def create_init_files():
    """Создает __init__.py файлы"""
    init_files = [
        "app/__init__.py",
        "app/models/__init__.py",
        "app/services/__init__.py",
        "app/api/__init__.py",
        "app/api/endpoints/__init__.py",
        "app/core/__init__.py",
        "app/utils/__init__.py",
        "tests/__init__.py",
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('"""Package initialization"""\n')
        print(f"✓ Создан файл: {init_file}")


def create_gitignore():
    """Создает .gitignore"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data
data/
*.db
*.db-journal

# Env
.env
.env.local

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    print("✓ Создан .gitignore")


def create_env_file():
    """Создает .env файл с примерами"""
    env_content = """
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Database
DATABASE_PATH=./data/rag_data.db

# Vector Databases
CHROMA_PERSIST_DIR=./data/chroma_langchain_db
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Models
TRANSFORMERS_CACHE=./data/transformers_models
DEFAULT_EMBEDDING_MODEL=DeepVk/USER-bge-m3
DEFAULT_DEVICE=cpu

# File Upload
MAX_UPLOAD_SIZE=50
ALLOWED_EXTENSIONS=pdf,txt,md
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_content.strip())
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content.strip())
        print("✓ Создан .env файл")
    else:
        print("⚠ .env уже существует, пропускаем")
    
    print("✓ Создан .env.example")


def create_requirements():
    """Создает requirements.txt"""
    requirements = """
# FastAPI
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
python-dotenv==1.0.0

# LangChain
langchain==0.1.0
langchain-community==0.0.10
langchain-text-splitters==0.0.1
langchain-huggingface==0.0.1
langchain-chroma==0.1.0

# ML/AI
sentence-transformers==2.2.2
chromadb==0.4.22
huggingface-hub==0.19.4

# PDF processing
pypdf==3.17.4

# Async
aiofiles==23.2.1
httpx==0.25.2

# Testing (optional)
pytest==7.4.3
pytest-asyncio==0.21.1
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements.strip())
    print("✓ Создан requirements.txt")


def create_readme():
    """Создает README.md"""
    readme = """
# RAG Pipeline API

REST API для обработки документов в векторную базу данных.

## Установка

```bash
# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\\Scripts\\activate  # Windows

# Установить зависимости
pip install -r requirements.txt
```

## Запуск

```bash
# Development
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Документация

После запуска доступна по адресу:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Структура проекта

```
app/
├── main.py              # FastAPI приложение
├── models/              # Pydantic модели
│   ├── configs.py       # Конфигурации компонентов
│   └── schemas.py       # API схемы
├── services/            # Бизнес-логика
│   ├── database.py      # Работа с БД
│   └── pipeline_service.py  # Обработка документов
└── utils/               # Утилиты
```

## API Endpoints

- `GET /api/loaders` - Список доступных загрузчиков
- `GET /api/splitters` - Список разделителей текста
- `GET /api/embeddings` - Список моделей эмбеддингов
- `GET /api/databases` - Список векторных БД
- `POST /api/pipelines` - Создать пайплайн
- `POST /api/pipelines/{id}/process` - Обработать файл
- `GET /api/tasks/{id}` - Статус задачи

## Примеры использования

См. документацию в `/docs` после запуска сервера.
"""
    
    with open('README.md', 'w') as f:
        f.write(readme.strip())
    print("✓ Создан README.md")


def create_run_script():
    """Создает скрипт запуска"""
    run_script = """#!/usr/bin/env python3
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
"""
    
    with open('run.py', 'w') as f:
        f.write(run_script.strip())
    
    # Make executable on Unix
    if os.name != 'nt':
        os.chmod('run.py', 0o755)
    
    print("✓ Создан run.py")


def verify_files_exist():
    """Проверяет наличие необходимых файлов"""
    required_files = [
        'app/main.py',
        'app/models/configs.py',
        'app/models/schemas.py',
        'app/services/database.py',
        'app/services/pipeline_service.py',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return missing_files


def main():
    print("\n" + "="*60)
    print("  RAG Pipeline API - Настройка проекта")
    print("="*60 + "\n")
    
    # Создаем структуру
    print("📁 Создание структуры директорий...")
    create_directory_structure()
    print()
    
    print("📝 Создание __init__.py файлов...")
    create_init_files()
    print()
    
    print("⚙️  Создание конфигурационных файлов...")
    create_gitignore()
    create_env_file()
    create_requirements()
    create_readme()
    create_run_script()
    print()
    
    # Проверка
    print("🔍 Проверка наличия основных файлов...")
    missing = verify_files_exist()
    
    if missing:
        print("\n⚠️  ВНИМАНИЕ: Следующие файлы необходимо создать вручную:")
        for file_path in missing:
            print(f"   - {file_path}")
        print("\n📚 Используйте артефакты из Claude для создания этих файлов:")
        print("   - models_configs → app/models/configs.py")
        print("   - models_schemas → app/models/schemas.py")
        print("   - db_manager → app/services/database.py")
        print("   - pipeline_service → app/services/pipeline_service.py")
        print("   - fastapi_main → app/main.py")
    else:
        print("✓ Все основные файлы присутствуют")
    
    print("\n" + "="*60)
    print("✅ Настройка завершена!")
    print("="*60)
    print("\n📋 Следующие шаги:")
    print("   1. Создайте недостающие файлы (если есть)")
    print("   2. Установите зависимости: pip install -r requirements.txt")
    print("   3. Запустите сервер: python run.py")
    print("   4. Откройте: http://localhost:8000/docs")
    print()


if __name__ == "__main__":
    main()