# Инструкция: Добавление режима выбора существующего пайплайна

## Текущее состояние
Сейчас `create_pipeline.html` позволяет только создавать новый пайплайн с вариантом.

## Требуемый функционал
Добавить возможность:
1. Выбрать существующий пайплайн
2. Выбрать существующий вариант ИЛИ создать новый вариант для этого пайплайна
3. Обработать файл с выбранной комбинацией

## Изменения в HTML

### 1. Добавить переключатель режимов (в начале content)

```html
<!-- Mode Selection -->
<div class="section">
    <h2>🎯 Режим работы</h2>
    <div class="mode-selector">
        <div class="mode-card active" id="modeNew" onclick="selectMode('new')">
            <h3>✨ Создать новый</h3>
            <p>Создать новый пайплайн с вариантом обработки</p>
        </div>
        <div class="mode-card" id="modeExisting" onclick="selectMode('existing')">
            <h3>📚 Использовать существующий</h3>
            <p>Выбрать готовый пайплайн и вариант</p>
        </div>
    </div>
</div>
```

### 2. Добавить секцию выбора существующего пайплайна

```html
<!-- Existing Pipeline Selection (hidden by default) -->
<div class="section" id="existingPipelineSection" style="display: none;">
    <h2>📚 Выбор пайплайна и варианта</h2>
    
    <!-- Pipeline List -->
    <div class="form-group">
        <label>Доступные пайплайны</label>
        <div id="pipelineList" class="pipeline-list">
            <div class="empty-state">Загрузка...</div>
        </div>
    </div>
    
    <!-- Variant Selection (shown after pipeline selected) -->
    <div id="variantSelection" style="display: none; margin-top: 20px;">
        <label>Варианты обработки</label>
        <div id="variantList" class="variant-list"></div>
        <button class="create-new-variant-btn" onclick="showNewVariantForm()">
            + Создать новый вариант
        </button>
    </div>
    
    <!-- New Variant Form (shown when creating new variant) -->
    <div id="newVariantForm" style="display: none; margin-top: 20px;">
        <h3>Создание нового варианта</h3>
        <div class="form-group">
            <label>Название варианта</label>
            <input type="text" id="newVariantName" placeholder="Например: PDF Processor">
        </div>
        <div class="form-group">
            <label>Описание</label>
            <input type="text" id="newVariantDescription" placeholder="Опционально">
        </div>
        <!-- Loader and Splitter selectors here -->
    </div>
</div>
```

### 3. Обернуть существующую форму создания в div

```html
<!-- New Pipeline Creation (shown by default) -->
<div id="newPipelineSection">
    <!-- Весь существующий код создания пайплайна -->
</div>
```

## JavaScript функции

### 1. Переключение режимов

```javascript
let currentMode = 'new';  // 'new' or 'existing'
let selectedPipelineId = null;
let selectedVariantId = null;

function selectMode(mode) {
    currentMode = mode;
    
    // Update UI
    document.getElementById('modeNew').classList.toggle('active', mode === 'new');
    document.getElementById('modeExisting').classList.toggle('active', mode === 'existing');
    
    // Show/hide sections
    document.getElementById('newPipelineSection').style.display = mode === 'new' ? 'block' : 'none';
    document.getElementById('existingPipelineSection').style.display = mode === 'existing' ? 'block' : 'none';
    
    // Load pipelines if switching to existing mode
    if (mode === 'existing') {
        loadExistingPipelines();
    }
    
    // Update button text
    updateProcessButtonText();
}
```

### 2. Загрузка существующих пайплайнов

```javascript
async function loadExistingPipelines() {
    try {
        const response = await fetch(`${API_URL}/api/pipelines`);
        const pipelines = await response.json();
        
        const listContainer = document.getElementById('pipelineList');
        
        if (pipelines.length === 0) {
            listContainer.innerHTML = '<div class="empty-state">Нет доступных пайплайнов</div>';
            return;
        }
        
        listContainer.innerHTML = pipelines.map(p => `
            <div class="pipeline-item" onclick="selectPipeline('${p.id}')">
                <h4>${p.name}</h4>
                <div class="pipeline-meta">
                    ID: ${p.id.substring(0, 8)}... | 
                    Создан: ${new Date(p.created_at).toLocaleDateString()}
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading pipelines:', error);
        showStatus('Ошибка загрузки пайплайнов', 'error');
    }
}
```

### 3. Выбор пайплайна и загрузка вариантов

```javascript
async function selectPipeline(pipelineId) {
    selectedPipelineId = pipelineId;
    
    // Update UI
    document.querySelectorAll('.pipeline-item').forEach(item => {
        item.classList.remove('selected');
    });
    event.target.closest('.pipeline-item').classList.add('selected');
    
    // Load variants for this pipeline
    try {
        const response = await fetch(`${API_URL}/api/pipelines/${pipelineId}`);
        const pipelineData = await response.json();
        
        const variantSelection = document.getElementById('variantSelection');
        const variantList = document.getElementById('variantList');
        
        if (!pipelineData.variants || pipelineData.variants.length === 0) {
            variantList.innerHTML = '<div class="empty-state">Нет вариантов. Создайте новый.</div>';
        } else {
            variantList.innerHTML = pipelineData.variants.map(v => `
                <div class="variant-item" onclick="selectVariant('${v.variant_id}')">
                    <strong>${v.name}</strong>
                    <div style="font-size: 0.85em; color: #666;">
                        ${v.description || 'Без описания'} | 
                        Файлов: ${v.files_processed || 0}
                    </div>
                </div>
            `).join('');
        }
        
        variantSelection.style.display = 'block';
    } catch (error) {
        console.error('Error loading variants:', error);
        showStatus('Ошибка загрузки вариантов', 'error');
    }
}
```

### 4. Выбор варианта

```javascript
function selectVariant(variantId) {
    selectedVariantId = variantId;
    
    // Update UI
    document.querySelectorAll('.variant-item').forEach(item => {
        item.classList.remove('selected');
    });
    event.target.closest('.variant-item').classList.add('selected');
    
    // Hide new variant form if shown
    document.getElementById('newVariantForm').style.display = 'none';
    
    updateProcessButtonText();
}
```

### 5. Показать форму создания нового варианта

```javascript
function showNewVariantForm() {
    const form = document.getElementById('newVariantForm');
    form.style.display = form.style.display === 'none' ? 'block' : 'none';
    
    // Deselect any selected variant
    selectedVariantId = null;
    document.querySelectorAll('.variant-item').forEach(item => {
        item.classList.remove('selected');
    });
}
```

### 6. Обновить текст кнопки обработки

```javascript
function updateProcessButtonText() {
    const btn = document.getElementById('processBtn');
    
    if (currentMode === 'new') {
        btn.textContent = 'Создать пайплайн и обработать файл';
    } else {
        if (selectedVariantId) {
            btn.textContent = 'Обработать файл с выбранным вариантом';
        } else if (document.getElementById('newVariantForm').style.display !== 'none') {
            btn.textContent = 'Создать вариант и обработать файл';
        } else {
            btn.textContent = 'Выберите вариант или создайте новый';
        }
    }
}
```

### 7. Обновить функцию processPipeline()

```javascript
async function processPipeline() {
    const file = document.getElementById('fileInput').files[0];
    if (!file) {
        alert('Пожалуйста, выберите файл');
        return;
    }
    
    const btn = document.getElementById('processBtn');
    btn.disabled = true;
    
    try {
        let pipelineId, variantId;
        
        if (currentMode === 'new') {
            // Existing logic for creating new pipeline
            // ... (keep existing code)
            
        } else {
            // Existing pipeline mode
            if (!selectedPipelineId) {
                alert('Выберите пайплайн');
                btn.disabled = false;
                return;
            }
            
            pipelineId = selectedPipelineId;
            
            // Check if creating new variant or using existing
            if (document.getElementById('newVariantForm').style.display !== 'none') {
                // Create new variant
                const variantConfig = {
                    name: document.getElementById('newVariantName').value,
                    description: document.getElementById('newVariantDescription').value,
                    loader: {
                        type: document.getElementById('loaderType').value,
                        ...currentParams.loader
                    },
                    splitter: {
                        type: document.getElementById('splitterType').value,
                        ...currentParams.splitter
                    }
                };
                
                const variantResp = await fetch(
                    `${API_URL}/api/pipelines/${pipelineId}/variants`,
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(variantConfig)
                    }
                );
                
                if (!variantResp.ok) {
                    throw new Error('Ошибка создания варианта');
                }
                
                const variantData = await variantResp.json();
                variantId = variantData.variant_id;
                
            } else if (selectedVariantId) {
                // Use selected variant
                variantId = selectedVariantId;
            } else {
                alert('Выберите вариант или создайте новый');
                btn.disabled = false;
                return;
            }
        }
        
        // Upload and process file (same for both modes)
        btn.textContent = 'Загрузка файла...';
        const formData = new FormData();
        formData.append('file', file);
        
        const uploadUrl = `${API_URL}/api/pipelines/${pipelineId}/process?variant_id=${variantId}`;
        const uploadResp = await fetch(uploadUrl, {
            method: 'POST',
            body: formData
        });
        
        if (!uploadResp.ok) {
            const error = await uploadResp.json();
            throw new Error(error.detail || 'Ошибка загрузки файла');
        }
        
        const uploadData = await uploadResp.json();
        const taskId = uploadData.task_id;
        
        // Show task info and poll status
        // ... (existing code)
        
    } catch (error) {
        console.error('Processing error:', error);
        showStatus(`Ошибка: ${error.message}`, 'error');
        btn.disabled = false;
        btn.textContent = updateProcessButtonText();
    }
}
```

## CSS стили

```css
.mode-selector {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    margin-bottom: 20px;
}

.mode-card {
    padding: 20px;
    border: 2px solid #e0e0e0;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.3s;
    text-align: center;
}

.mode-card:hover {
    border-color: #667eea;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
}

.mode-card.active {
    border-color: #667eea;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
}

.pipeline-list {
    max-height: 300px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px;
}

.pipeline-item {
    padding: 15px;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    margin-bottom: 10px;
    cursor: pointer;
    transition: all 0.3s;
}

.pipeline-item:hover {
    border-color: #667eea;
    background: #f8f9fa;
}

.pipeline-item.selected {
    border-color: #667eea;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
}

.variant-item {
    padding: 10px;
    background: white;
    border-left: 3px solid #ff6b6b;
    margin-bottom: 5px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
}

.variant-item:hover {
    background: #fff5f5;
}

.variant-item.selected {
    background: #ffe0e0;
    border-left-width: 5px;
}

.create-new-variant-btn {
    background: #ff6b6b;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9em;
    margin-top: 10px;
}

.create-new-variant-btn:hover {
    background: #ff5252;
}
```

## Итоговый workflow

### Режим "Создать новый"
1. Пользователь выбирает все компоненты (embedding, database, loader, splitter)
2. Загружает файл
3. Создается pipeline с default_variant
4. Файл обрабатывается

### Режим "Использовать существующий"
1. Пользователь выбирает существующий pipeline из списка
2. Видит список вариантов для этого pipeline
3. Либо выбирает существующий вариант
4. Либо создает новый вариант (выбирает loader + splitter)
5. Загружает файл
6. Файл обрабатывается с выбранным/созданным вариантом

## Преимущества
- ✅ Переиспользование существующих баз знаний
- ✅ Возможность добавлять новые варианты обработки
- ✅ Гибкость в выборе комбинации pipeline + variant
- ✅ Интуитивный интерфейс с визуальным разделением
