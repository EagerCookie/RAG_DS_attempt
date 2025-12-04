#!/usr/bin/env python3
"""
Скрипт для проверки переменных окружения
Использование: python check_env.py
"""

import os
from dotenv import load_dotenv

print("="*60)
print("🔍 Проверка переменных окружения")
print("="*60)
print()

# Загрузка .env файла
env_file = ".env"
if os.path.exists(env_file):
    print(f"✓ Файл {env_file} найден")
    load_dotenv()
    print(f"✓ Переменные загружены из {env_file}")
else:
    print(f"❌ Файл {env_file} не найден")
    print(f"   Создайте файл .env в корне проекта")

print()
print("─"*60)
print("Проверка API ключей:")
print("─"*60)

# Проверка ключей
keys_to_check = {
    "OPENAI_API_KEY": "OpenAI",
    "ANTHROPIC_API_KEY": "Anthropic (опционально)",
    "DEEPSEEK_API_KEY": "DeepSeek (опционально)",
}

found_keys = []
missing_keys = []

for key, description in keys_to_check.items():
    value = os.getenv(key)
    
    if value:
        # Маскируем ключ для безопасности
        masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
        print(f"✓ {key}: {masked}")
        print(f"  └─ {description}")
        found_keys.append(key)
    else:
        print(f"✗ {key}: не найден")
        print(f"  └─ {description}")
        missing_keys.append(key)
    print()

# Итоги
print("="*60)
if "OPENAI_API_KEY" in found_keys:
    print("✅ OpenAI ключ настроен - можно использовать RAG inference")
else:
    print("❌ OpenAI ключ не найден - добавьте в .env:")
    print()
    print("   OPENAI_API_KEY=sk-...")
    print()

if len(found_keys) > 0:
    print(f"\n✓ Найдено ключей: {len(found_keys)}")
    print(f"  Провайдеры: {', '.join([k.replace('_API_KEY', '') for k in found_keys])}")

if len(missing_keys) > 0:
    print(f"\n⚠ Отсутствует ключей: {len(missing_keys)}")
    print(f"  (опциональные: Anthropic, DeepSeek)")

print()

# Проверка других важных переменных
print("─"*60)
print("Другие переменные:")
print("─"*60)

other_vars = [
    "DATABASE_PATH",
    "CHROMA_PERSIST_DIR",
    "TRANSFORMERS_CACHE",
]

for var in other_vars:
    value = os.getenv(var)
    if value:
        print(f"✓ {var}: {value}")
    else:
        print(f"✗ {var}: не задан (будет использоваться default)")

print()
print("="*60)

# Тест импорта
print("\n🔧 Проверка зависимостей:")
print("─"*60)

packages = {
    "langchain_openai": "OpenAI интеграция",
    "langchain_anthropic": "Anthropic интеграция (опционально)",
    "langchain_chroma": "Chroma векторная БД",
    "langchain_huggingface": "HuggingFace эмбеддинги",
}

for package, description in packages.items():
    try:
        __import__(package)
        print(f"✓ {package}: установлен")
    except ImportError:
        print(f"✗ {package}: не установлен")
        print(f"  └─ {description}")
        if package in ["langchain_openai", "langchain_chroma", "langchain_huggingface"]:
            print(f"  └─ pip install {package.replace('_', '-')}")

print()
print("="*60)
print("✅ Проверка завершена")
print("="*60)
print()

# Рекомендации
if "OPENAI_API_KEY" not in found_keys:
    print("📋 Следующие шаги:")
    print("1. Создайте/отредактируйте файл .env в корне проекта")
    print("2. Добавьте строку: OPENAI_API_KEY=sk-ваш-ключ")
    print("3. Перезапустите сервер: uvicorn app.main:app --reload")
    print("4. Попробуйте снова")
else:
    print("🎉 Всё готово для использования RAG inference!")
    print()
    print("Запустите сервер, если ещё не запущен:")
    print("  uvicorn app.main:app --reload")
    print()
    print("Откройте rag_inference.html в браузере")