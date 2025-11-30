#!/usr/bin/env python3
"""
Инспектор Chroma векторной базы данных
Использование: python inspect_chroma.py
"""

import chromadb
from chromadb.config import Settings
import os
import sys


def inspect_chroma_database(persist_directory="./chroma_langchain_db"):
    """
    Инспектирует Chroma базу данных и выводит информацию
    
    Args:
        persist_directory: путь к директории с Chroma БД
    """
    print("="*60)
    print("🔍 Chroma Database Inspector")
    print("="*60)
    print()
    
    # Проверка существования директории
    if not os.path.exists(persist_directory):
        print(f"❌ Директория не найдена: {persist_directory}")
        print(f"   Проверьте путь или запустите обработку пайплайна")
        return
    
    print(f"📁 Директория БД: {persist_directory}")
    print()
    
    try:
        # Подключение к Chroma
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Получение списка всех коллекций
        collections = client.list_collections()
        
        print(f"📊 Найдено коллекций: {len(collections)}")
        print()
        
        if not collections:
            print("⚠️  Коллекций не найдено. База данных пуста.")
            return
        
        # Инспектирование каждой коллекции
        for idx, collection in enumerate(collections, 1):
            print("─"*60)
            print(f"📦 Коллекция #{idx}: {collection.name}")
            print("─"*60)
            
            # Получение метаданных коллекции
            print(f"   ID: {collection.id}")
            print(f"   Metadata: {collection.metadata}")
            print()
            
            # Получение количества документов
            count = collection.count()
            print(f"   📄 Документов (chunks): {count}")
            
            if count == 0:
                print("   ⚠️  Коллекция пуста")
                continue
            
            # Получение примера документов (первые 5)
            results = collection.get(
                limit=5,
                include=['embeddings', 'documents', 'metadatas']
            )
            
            # Информация о размерности эмбеддингов
            if results['embeddings'] is not None and len(results['embeddings']) > 0:
                embedding_dim = len(results['embeddings'][0])
                print(f"   🧠 Размерность эмбеддингов: {embedding_dim}")
            
            print()
            print("   📝 Примеры документов (первые 5):")
            print()
            
            for i, doc_id in enumerate(results['ids']):
                print(f"   [{i+1}] ID: {doc_id}")
                
                # Документ (текст)
                if results['documents'] is not None and i < len(results['documents']):
                    doc_text = results['documents'][i]
                    if doc_text:
                        preview = doc_text[:100] + "..." if len(doc_text) > 100 else doc_text
                        print(f"       Текст: {preview}")
                
                # Метаданные
                if results['metadatas'] is not None and i < len(results['metadatas']):
                    metadata = results['metadatas'][i]
                    if metadata:
                        print(f"       Metadata: {metadata}")
                
                # Эмбеддинг (первые 5 значений)
                if results['embeddings'] is not None and i < len(results['embeddings']):
                    embedding = results['embeddings'][i]
                    if embedding is not None and len(embedding) > 0:
                        preview_emb = embedding[:5]
                        print(f"       Embedding: [{', '.join(f'{x:.4f}' for x in preview_emb)}, ...]")
                
                print()
            
            # Статистика по метаданным
            print("   📊 Статистика метаданных:")
            if results['metadatas'] is not None and len(results['metadatas']) > 0:
                # Собираем все ключи метаданных
                all_keys = set()
                for metadata in results['metadatas']:
                    if metadata:
                        all_keys.update(metadata.keys())
                
                if all_keys:
                    print(f"       Поля метаданных: {', '.join(all_keys)}")
                else:
                    print("       Метаданные отсутствуют")
            else:
                print("       Метаданные отсутствуют")
            
            print()
        
        print("="*60)
        print("✅ Инспекция завершена")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Ошибка при инспекции: {e}")
        import traceback
        traceback.print_exc()


def test_search(persist_directory="./chroma_langchain_db", collection_name=None, query_text="test"):
    """
    Тестовый поиск в коллекции
    
    Args:
        persist_directory: путь к БД
        collection_name: имя коллекции (если None, использует первую)
        query_text: текст для поиска
    """
    print("\n" + "="*60)
    print("🔎 Тест поиска в векторной БД")
    print("="*60)
    print()
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Подключение к Chroma
        client = chromadb.PersistentClient(path=persist_directory)
        collections = client.list_collections()
        
        if not collections:
            print("❌ Нет доступных коллекций для поиска")
            return
        
        # Выбор коллекции
        if collection_name:
            collection = client.get_collection(name=collection_name)
        else:
            collection = collections[0]
        
        print(f"📦 Коллекция: {collection.name}")
        print(f"🔍 Поисковый запрос: '{query_text}'")
        print()
        
        # Создание эмбеддинга для запроса
        # ВАЖНО: Используйте ту же модель, что и при создании БД!
        embeddings = HuggingFaceEmbeddings(
            model_name="DeepVk/USER-bge-m3",  # Замените на вашу модель
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder="./transformers_models"
        )
        
        print("⏳ Создание эмбеддинга для запроса...")
        query_embedding = embeddings.embed_query(query_text)
        print(f"✓ Эмбеддинг создан (размерность: {len(query_embedding)})")
        print()
        
        # Поиск
        print("⏳ Выполнение поиска (top 3)...")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"✓ Найдено результатов: {len(results['ids'][0])}")
        print()
        
        # Вывод результатов
        for i, doc_id in enumerate(results['ids'][0]):
            distance = results['distances'][0][i] if results['distances'] else None
            doc_text = results['documents'][0][i] if results['documents'] else ""
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            
            print(f"🎯 Результат #{i+1}")
            print(f"   ID: {doc_id}")
            print(f"   Distance: {distance:.4f}" if distance is not None else "   Distance: N/A")
            print(f"   Metadata: {metadata}")
            print(f"   Текст: {doc_text[:200]}...")
            print()
        
    except Exception as e:
        print(f"❌ Ошибка при поиске: {e}")
        import traceback
        traceback.print_exc()


def list_all_subdirectories(persist_directory="./chroma_langchain_db"):
    """
    Показывает все поддиректории (каждая = отдельная БД пайплайна)
    """
    print("\n" + "="*60)
    print("📂 Список векторных баз (по пайплайнам)")
    print("="*60)
    print()
    
    if not os.path.exists(persist_directory):
        print(f"❌ Директория не найдена: {persist_directory}")
        return []
    
    subdirs = []
    for item in os.listdir(persist_directory):
        item_path = os.path.join(persist_directory, item)
        if os.path.isdir(item_path):
            subdirs.append(item_path)
    
    if not subdirs:
        print("⚠️  Поддиректории не найдены")
        return []
    
    print(f"Найдено {len(subdirs)} векторных баз:")
    print()
    
    for idx, subdir in enumerate(subdirs, 1):
        print(f"{idx}. {os.path.basename(subdir)}")
        
        # Попытка получить информацию о коллекциях
        try:
            client = chromadb.PersistentClient(path=subdir)
            collections = client.list_collections()
            
            for coll in collections:
                count = coll.count()
                print(f"   └─ Коллекция: {coll.name} ({count} документов)")
        except:
            print(f"   └─ (не удалось прочитать)")
    
    print()
    return subdirs


def interactive_mode():
    """Интерактивный режим выбора действия"""
    print("\n" + "="*60)
    print("🎮 Интерактивный режим")
    print("="*60)
    print()
    print("Выберите действие:")
    print("1. Показать все векторные базы")
    print("2. Инспектировать конкретную базу")
    print("3. Тест поиска")
    print("4. Выход")
    print()
    
    choice = input("Ваш выбор (1-4): ").strip()
    
    if choice == "1":
        subdirs = list_all_subdirectories()
        if subdirs:
            input("\nНажмите Enter для продолжения...")
            interactive_mode()
    
    elif choice == "2":
        subdirs = list_all_subdirectories()
        if subdirs:
            idx = input(f"\nВыберите базу (1-{len(subdirs)}): ").strip()
            try:
                selected = subdirs[int(idx) - 1]
                inspect_chroma_database(selected)
            except (ValueError, IndexError):
                print("❌ Неверный выбор")
            input("\nНажмите Enter для продолжения...")
            interactive_mode()
    
    elif choice == "3":
        subdirs = list_all_subdirectories()
        if subdirs:
            idx = input(f"\nВыберите базу (1-{len(subdirs)}): ").strip()
            query = input("Введите поисковый запрос: ").strip()
            try:
                selected = subdirs[int(idx) - 1]
                test_search(selected, query_text=query or "test")
            except (ValueError, IndexError):
                print("❌ Неверный выбор")
            input("\nНажмите Enter для продолжения...")
            interactive_mode()
    
    elif choice == "4":
        print("👋 До свидания!")
        sys.exit(0)
    
    else:
        print("❌ Неверный выбор")
        interactive_mode()


def main():
    """Главная функция"""
    print("\n🚀 Chroma Database Inspector")
    print()
    
    # Проверка аргументов командной строки
    if len(sys.argv) > 1:
        persist_directory = sys.argv[1]
        print(f"📁 Использую путь: {persist_directory}")
        inspect_chroma_database(persist_directory)
        
        # Опционально: тест поиска
        if len(sys.argv) > 2:
            query = sys.argv[2]
            test_search(persist_directory, query_text=query)
    else:
        # Интерактивный режим
        interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Прервано пользователем")
        sys.exit(0)