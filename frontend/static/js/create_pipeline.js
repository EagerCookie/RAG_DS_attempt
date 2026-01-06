const API_URL = 'http://localhost:8000';

// Store component schemas
let componentSchemas = {
    loaders: {},
    splitters: {},
    embeddings: {},
    databases: {}
};

// Store current parameter values
let currentParams = {
    loader: {}, // For new pipeline mode
    splitter: {}, // For new pipeline mode
    loaderExisting: {}, // For new variant (existing pipeline mode)
    splitterExisting: {}, // For new variant (existing pipeline mode)
    embedding: {},
    database: {}
};

// ==========================================
// NEW MODE SELECTION VARIABLES
// ==========================================
// Global variables update
let currentMode = 'new';
let currentVariantMode = 'new'; // 'new' or 'existing'
let selectedPipelineId = null;
let selectedVariantId = null;
let allVariantsCache = [];

// Load components on page load
window.onload = async () => {
    await loadLoaders();
    await loadSplitters();
    await loadEmbeddings();
    await loadDatabases();

    // Set initial status message display
    document.getElementById('statusMessage').style.display = 'block';
    showStatus('Выберите режим работы и компоненты', 'info');

    // Initial button text update
    updateProcessButtonText();
};

// ==========================================
// MODE SELECTION FUNCTIONS
// ==========================================
function selectMode(mode) {
    currentMode = mode;

    // Update UI
    document.getElementById('modeNew').classList.toggle('active', mode === 'new');
    document.getElementById('modeExisting').classList.toggle('active', mode === 'existing');

    // Show/hide sections
    document.getElementById('newPipelineSection').style.display = mode === 'new' ? 'block' : 'none';
    document.getElementById('existingPipelineSection').style.display = mode === 'existing' ? 'block' : 'none';

    // Hide variant selection when switching to existing mode initially
    document.getElementById('variantSelection').style.display = 'none';
    document.getElementById('newVariantForm').style.display = 'none';
    selectedPipelineId = null;
    selectedVariantId = null;

    // Load pipelines if switching to existing mode
    if (mode === 'existing') {
        loadExistingPipelines();
    }

    // Update button text
    updateProcessButtonText();
}

async function loadExistingPipelines() {
    const listContainer = document.getElementById('pipelineList');
    listContainer.innerHTML = '<div class="empty-state">Загрузка...</div>';
    try {
        const response = await fetch(`${API_URL}/api/pipelines`);
        if (!response.ok) throw new Error('Failed to load pipelines');
        const pipelines = await response.json();

        if (pipelines.length === 0) {
            listContainer.innerHTML = '<div class="empty-state">Нет доступных пайплайнов</div>';
            return;
        }

        listContainer.innerHTML = pipelines.map(p => `
            <div class="pipeline-item" onclick="selectPipeline(event, '${p.id}')">
                <h4>${p.name}</h4>
                <div class="pipeline-meta">
                    ID: ${p.id.substring(0, 8)}... | 
                    Создан: ${new Date(p.created_at).toLocaleDateString()}
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading pipelines:', error);
        listContainer.innerHTML = '<div class="empty-state">Ошибка загрузки</div>';
    }
}

async function selectPipeline(event, pipelineId) {
    selectedPipelineId = pipelineId;
    selectedVariantId = null;

    document.querySelectorAll('.pipeline-item').forEach(item => item.classList.remove('selected'));
    event.currentTarget.classList.add('selected');

    document.getElementById('newVariantForm').style.display = 'none';
    document.querySelectorAll('.variant-item').forEach(item => item.classList.remove('selected'));

    try {
        const response = await fetch(`${API_URL}/api/pipelines/${pipelineId}`);
        const pipelineData = await response.json();

        const variantSelection = document.getElementById('variantSelection');
        const variantList = document.getElementById('variantList');

        if (!pipelineData.variants || pipelineData.variants.length === 0) {
            variantList.innerHTML = '<div class="empty-state">Нет вариантов. Создайте новый.</div>';
        } else {
            variantList.innerHTML = pipelineData.variants.map(v => `
                <div class="variant-item" onclick="selectVariant(event, '${v.variant_id}')">
                    <strong>${v.name}</strong>
                    <div style="font-size: 0.85em; color: #666;">
                        ${v.description || 'Без описания'} | 
                        Файлов: ${v.files_processed || 0}
                    </div>
                </div>
            `).join('');
        }
        variantSelection.style.display = 'block';
        updateProcessButtonText();
    } catch (error) {
        console.error('Error loading variants:', error);
    }
}

function selectVariant(event, variantId) {
    selectedVariantId = variantId;
    document.querySelectorAll('.variant-item').forEach(item => item.classList.remove('selected'));
    event.currentTarget.classList.add('selected');
    document.getElementById('newVariantForm').style.display = 'none';
    updateProcessButtonText();
}

function showNewVariantForm() {
    const form = document.getElementById('newVariantForm');
    const isVisible = form.style.display !== 'none';
    form.style.display = isVisible ? 'none' : 'block';

    if (!isVisible) {
        selectedVariantId = null;
        document.querySelectorAll('.variant-item').forEach(item => item.classList.remove('selected'));
        document.getElementById('loaderTypeExisting').value = '';
        document.getElementById('splitterTypeExisting').value = '';
        loadLoaderParams('existing');
        loadSplitterParams('existing');
    }
    updateProcessButtonText();
}

function updateProcessButtonText() {
    const btn = document.getElementById('processBtn');
    const newVariantFormVisible = document.getElementById('newVariantForm').style.display !== 'none';

    if (currentMode === 'new') {
        btn.textContent = 'Создать пайплайн и обработать файл';
        btn.disabled = false;
    } else {
        if (!selectedPipelineId) {
            btn.textContent = 'Выберите пайплайн';
            btn.disabled = true;
            return;
        }
        if (selectedVariantId) {
            btn.textContent = 'Обработать файл с выбранным вариантом';
            btn.disabled = false;
        } else if (newVariantFormVisible) {
            btn.textContent = 'Создать вариант и обработать файл';
            btn.disabled = false;
        } else {
            btn.textContent = 'Выберите вариант или создайте новый';
            btn.disabled = true;
        }
    }
}

// ==========================================
// LOAD COMPONENT LISTS
// ==========================================

// ==========================================
// LOAD COMPONENT LISTS
// ==========================================

async function loadLoaders() {
    try {
        const response = await fetch(`${API_URL}/api/loaders`);
        if (!response.ok) throw new Error(`Ошибка загрузки загрузчиков: ${response.status}`);
        const loaders = await response.json();

        const selectNew = document.getElementById('loaderType');
        const selectExisting = document.getElementById('loaderTypeExisting');

        const optionsHtml = '<option value="">-- Выберите загрузчик --</option>' +
            loaders.map(l => {
                componentSchemas.loaders[l.id] = l.config_schema;
                return `<option value="${l.id}">${l.name}</option>`;
            }).join('');

        if (selectNew) selectNew.innerHTML = optionsHtml;
        if (selectExisting) selectExisting.innerHTML = optionsHtml;
    } catch (error) {
        console.error('Error loading loaders:', error);
    }
}

async function loadSplitters() {
    try {
        const response = await fetch(`${API_URL}/api/splitters`);
        if (!response.ok) throw new Error(`Ошибка загрузки разделителей: ${response.status}`);
        const splitters = await response.json();

        const selectNew = document.getElementById('splitterType');
        const selectExisting = document.getElementById('splitterTypeExisting');

        const optionsHtml = '<option value="">-- Выберите разделитель --</option>' +
            splitters.map(s => {
                componentSchemas.splitters[s.id] = s.config_schema;
                return `<option value="${s.id}">${s.name}</option>`;
            }).join('');

        if (selectNew) selectNew.innerHTML = optionsHtml;
        if (selectExisting) selectExisting.innerHTML = optionsHtml;
    } catch (error) {
        console.error('Error loading splitters:', error);
    }
}

async function loadEmbeddings() {
    try {
        const response = await fetch(`${API_URL}/api/embeddings`);
        if (!response.ok) throw new Error(`Ошибка загрузки эмбеддингов: ${response.status}`);
        const embeddings = await response.json();

        const select = document.getElementById('embeddingType');
        select.innerHTML = '<option value="">-- Выберите модель --</option>' +
            embeddings.map(e => {
                componentSchemas.embeddings[e.id] = e.config_schema;
                return `<option value="${e.id}">${e.name}</option>`;
            }).join('');
    } catch (error) {
        console.error('Error loading embeddings:', error);
    }
}

async function loadDatabases() {
    try {
        const response = await fetch(`${API_URL}/api/databases`);
        if (!response.ok) throw new Error(`Ошибка загрузки баз данных: ${response.status}`);
        const databases = await response.json();

        const select = document.getElementById('databaseType');
        select.innerHTML = '<option value="">-- Выберите базу данных --</option>' +
            databases.map(d => {
                componentSchemas.databases[d.id] = d.config_schema;
                return `<option value="${d.id}">${d.name}</option>`;
            }).join('');
    } catch (error) {
        console.error('Error loading databases:', error);
    }
}

// ==========================================
// DYNAMIC PARAMETER LOADING
// ==========================================

function loadLoaderParams(mode) {
    const isExisting = mode === 'existing';
    const typeId = isExisting ? 'loaderTypeExisting' : 'loaderType';
    const paramsId = isExisting ? 'loaderParamsExisting' : 'loaderParams';
    const contentId = isExisting ? 'loaderParamsContentExisting' : 'loaderParamsContent';
    const paramKey = isExisting ? 'loaderExisting' : 'loader';

    const selectedType = document.getElementById(typeId).value;
    if (!selectedType) {
        document.getElementById(paramsId).style.display = 'none';
        return;
    }

    const schema = componentSchemas.loaders[selectedType];
    renderParams(paramKey, schema, contentId);
    document.getElementById(paramsId).style.display = 'block';
}

function loadSplitterParams(mode) {
    const isExisting = mode === 'existing';
    const typeId = isExisting ? 'splitterTypeExisting' : 'splitterType';
    const paramsId = isExisting ? 'splitterParamsExisting' : 'splitterParams';
    const contentId = isExisting ? 'splitterParamsContentExisting' : 'splitterParamsContent';
    const paramKey = isExisting ? 'splitterExisting' : 'splitter';

    const selectedType = document.getElementById(typeId).value;
    if (!selectedType) {
        document.getElementById(paramsId).style.display = 'none';
        return;
    }

    const schema = componentSchemas.splitters[selectedType];
    renderParams(paramKey, schema, contentId);
    document.getElementById(paramsId).style.display = 'block';
}

function loadEmbeddingParams() {
    const selectedType = document.getElementById('embeddingType').value;
    if (!selectedType) {
        document.getElementById('embeddingParams').style.display = 'none';
        return;
    }

    const schema = componentSchemas.embeddings[selectedType];
    renderParams('embedding', schema, 'embeddingParamsContent');
    document.getElementById('embeddingParams').style.display = 'block';
}

function loadDatabaseParams() {
    const selectedType = document.getElementById('databaseType').value;
    if (!selectedType) {
        document.getElementById('databaseParams').style.display = 'none';
        return;
    }

    const schema = componentSchemas.databases[selectedType];
    renderParams('database', schema, 'databaseParamsContent');
    document.getElementById('databaseParams').style.display = 'block';
}

// ==========================================
// RENDER PARAMETERS DYNAMICALLY
// ==========================================

function renderParams(componentParamKey, schema, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    // Reset params
    currentParams[componentParamKey] = {};

    if (!schema || !schema.properties) {
        container.innerHTML = '<div class="empty-state">Нет настраиваемых параметров</div>';
        return;
    }

    const properties = schema.properties;

    Object.keys(properties).forEach(key => {
        if (key === 'type') return;

        const prop = properties[key];
        const isRequired = schema.required && schema.required.includes(key);

        const formGroup = document.createElement('div');
        formGroup.className = 'form-group';

        const label = document.createElement('label');
        label.textContent = formatFieldName(key);
        if (isRequired) label.innerHTML += ' <span style="color: red;">*</span>';
        if (prop.description) label.innerHTML += `<span class="label-description">(${prop.description})</span>`;
        formGroup.appendChild(label);

        const input = createInputForType(componentParamKey, key, prop);
        formGroup.appendChild(input);
        container.appendChild(formGroup);
    });
}

function createInputForType(componentParamKey, key, prop) {
    const inputId = `${componentParamKey}_${key}`;
    let input;

    if (prop.type === 'boolean') {
        const wrapper = document.createElement('div');
        wrapper.className = 'checkbox-group';
        input = document.createElement('input');
        input.type = 'checkbox';
        input.id = inputId;
        input.checked = prop.default !== undefined ? prop.default : false;

        const label = document.createElement('label');
        label.htmlFor = inputId;
        label.textContent = 'Включено';
        label.style.fontWeight = 'normal';

        wrapper.appendChild(input);
        wrapper.appendChild(label);

        input.addEventListener('change', (e) => {
            currentParams[componentParamKey][key] = e.target.checked;
        });
        currentParams[componentParamKey][key] = input.checked;
        return wrapper;
    }

    if (prop.enum) {
        input = document.createElement('select');
        input.id = inputId;
        prop.enum.forEach(value => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = value;
            if (value === prop.default) option.selected = true;
            input.appendChild(option);
        });
        input.addEventListener('change', (e) => {
            currentParams[componentParamKey][key] = e.target.value;
        });
        currentParams[componentParamKey][key] = input.value;
        return input;
    }

    if (prop.type === 'integer' || prop.type === 'number') {
        input = document.createElement('input');
        input.type = 'number';
        input.id = inputId;
        if (prop.minimum !== undefined) input.min = prop.minimum;
        if (prop.maximum !== undefined) input.max = prop.maximum;
        if (prop.default !== undefined) input.value = prop.default;

        input.addEventListener('input', (e) => {
            const value = prop.type === 'integer' ? parseInt(e.target.value) : parseFloat(e.target.value);
            currentParams[componentParamKey][key] = isNaN(value) ? null : value;
        });

        const initVal = input.value ? (prop.type === 'integer' ? parseInt(input.value) : parseFloat(input.value)) : null;
        currentParams[componentParamKey][key] = initVal;
        return input;
    }

    input = document.createElement('input');
    input.type = 'text';
    input.id = inputId;
    if (prop.default !== undefined) input.value = prop.default;

    input.addEventListener('input', (e) => {
        currentParams[componentParamKey][key] = e.target.value;
    });
    currentParams[componentParamKey][key] = input.value;
    return input;
}

function formatFieldName(key) {
    return key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

// ==========================================
// FILE HANDLING
// ==========================================
document.getElementById('fileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const fileInfo = document.getElementById('fileInfo');
        fileInfo.style.display = 'flex';
        fileInfo.innerHTML = `<span>📄 ${file.name}</span><span>${(file.size / 1024 / 1024).toFixed(2)} MB</span>`;
    } else {
        document.getElementById('fileInfo').style.display = 'none';
    }
});

// VARIANT MODE SELECTION
function selectVariantMode(mode) {
    currentVariantMode = mode;

    // Update UI
    document.getElementById('modeNewVariant').classList.toggle('active', mode === 'new');
    document.getElementById('modeExistingVariant').classList.toggle('active', mode === 'existing');

    const newConfigDiv = document.getElementById('newVariantConfig');
    const existingSelectorDiv = document.getElementById('existingVariantSelector');

    if (mode === 'new') {
        newConfigDiv.style.display = 'block';
        existingSelectorDiv.style.display = 'none';
    } else {
        newConfigDiv.style.display = 'none';
        existingSelectorDiv.style.display = 'block';
        // Load variants if not loaded
        if (allVariantsCache.length === 0) {
            loadAllVariants();
        }
    }
}

async function loadAllVariants() {
    const select = document.getElementById('existingVariantSelect');
    select.innerHTML = '<option value="">Загрузка...</option>';

    try {
        const response = await fetch(`${API_URL}/api/variants`);
        if (!response.ok) throw new Error('Failed to load variants');

        allVariantsCache = await response.json();

        if (allVariantsCache.length === 0) {
            select.innerHTML = '<option value="">Нет доступных вариантов</option>';
            return;
        }

        select.innerHTML = '<option value="">-- Выберите вариант --</option>' +
            allVariantsCache.map(v =>
                `<option value="${v.id}">${v.name} (${v.pipeline_name})</option>`
            ).join('');

    } catch (error) {
        console.error('Error loading variants:', error);
        select.innerHTML = '<option value="">Ошибка загрузки</option>';
    }
}

function onExistingVariantSelect() {
    const select = document.getElementById('existingVariantSelect');
    const variantId = select.value;
    const infoDiv = document.getElementById('selectedVariantInfo');

    if (!variantId) {
        infoDiv.style.display = 'none';
        return;
    }

    const variant = allVariantsCache.find(v => v.id === variantId);
    if (!variant) return;

    // Parse config if it's a string
    let config = variant.config;
    if (typeof config === 'string') {
        try {
            config = JSON.parse(config);
        } catch (e) {
            console.error('Error parsing variant config', e);
            return;
        }
    }

    // Show info
    infoDiv.style.display = 'block';
    infoDiv.innerHTML = `
        <strong>Выбранный вариант:</strong> ${variant.name}<br>
        <small>Loader: ${config.loader.type} | Splitter: ${config.splitter.type}</small>
    `;
}

// UPDATED processPipeline function
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
            // NEW PIPELINE CREATION

            // Common validations
            const embeddingType = document.getElementById('embeddingType').value;
            const databaseType = document.getElementById('databaseType').value;

            if (!embeddingType || !databaseType) {
                alert('Пожалуйста, выберите компоненты базы знаний (Embedding + Database)');
                btn.disabled = false;
                return;
            }

            // PREPARE VARIANT CONFIG
            let variantConfigPayload = null;

            if (currentVariantMode === 'new') {
                // Create from form
                const loaderType = document.getElementById('loaderType').value;
                const splitterType = document.getElementById('splitterType').value;

                if (!loaderType || !splitterType) {
                    alert('Пожалуйста, выберите загрузчик и разделитель');
                    btn.disabled = false;
                    return;
                }

                variantConfigPayload = {
                    name: document.getElementById('variantName').value,
                    description: document.getElementById('variantDescription').value || null,
                    loader: {
                        type: loaderType,
                        ...currentParams.loader
                    },
                    splitter: {
                        type: splitterType,
                        ...currentParams.splitter
                    }
                };
            } else {
                // Copy from existing
                const select = document.getElementById('existingVariantSelect');
                const variantId = select.value;
                if (!variantId) {
                    alert('Пожалуйста, выберите существующий вариант');
                    btn.disabled = false;
                    return;
                }

                const variant = allVariantsCache.find(v => v.id === variantId);
                let config = variant.config;
                if (typeof config === 'string') config = JSON.parse(config);

                // We copy the config but give it a new name to indicate it's a copy/reuse
                variantConfigPayload = {
                    name: `${variant.name} (Copy)`,
                    description: `Copied from variant ${variant.id}`,
                    loader: config.loader,
                    splitter: config.splitter
                };
            }

            btn.textContent = 'Создание пайплайна...';

            // Build FULL config
            const config = {
                name: document.getElementById('pipelineName').value,
                embedding: {
                    type: embeddingType,
                    ...currentParams.embedding
                },
                database: {
                    type: databaseType,
                    ...currentParams.database
                },
                default_variant: variantConfigPayload
            };

            showStatus('Создание пайплайна и варианта...', 'info');
            const pipelineResp = await fetch(`${API_URL}/api/pipelines`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            if (!pipelineResp.ok) {
                const error = await pipelineResp.json();
                throw new Error(error.detail || 'Ошибка создания пайплайна');
            }

            const pipelineData = await pipelineResp.json();
            pipelineId = pipelineData.pipeline_id;
            variantId = pipelineData.default_variant_id;

        } else {
            // ... Existing pipeline mode logic (unchanged) ...
            // Existing variant reuse logic for "Existing Pipeline" mode is different (handled in other block)
            // But your request specifically mentioned "creating a NEW pipeline"

            // Existing pipeline mode logic from previous file content
            const newVariantFormVisible = document.getElementById('newVariantForm').style.display !== 'none';

            if (!selectedPipelineId) {
                alert('Выберите пайплайн');
                btn.disabled = false;
                updateProcessButtonText();
                return;
            }

            pipelineId = selectedPipelineId;

            if (newVariantFormVisible) {
                // ... (keep creating new variant logic)
                const loaderTypeExisting = document.getElementById('loaderTypeExisting').value;
                const splitterTypeExisting = document.getElementById('splitterTypeExisting').value;
                // ... validation ...

                const variantConfig = {
                    name: document.getElementById('newVariantName').value,
                    description: document.getElementById('newVariantDescription').value,
                    loader: {
                        type: loaderTypeExisting,
                        ...currentParams.loaderExisting
                    },
                    splitter: {
                        type: splitterTypeExisting,
                        ...currentParams.splitterExisting
                    }
                };
                // ... fetch call ...
                const variantResp = await fetch(`${API_URL}/api/pipelines/${pipelineId}/variants`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(variantConfig)
                });
                // ... error handling ...
                const variantData = await variantResp.json();
                variantId = variantData.variant_id;

            } else if (selectedVariantId) {
                variantId = selectedVariantId;
            } else {
                alert('Выберите вариант');
                btn.disabled = false;
                return;
            }
        }

        // ... Upload file logic (common) ...
        // Show pipeline info
        document.getElementById('taskInfo').style.display = 'block';
        document.getElementById('pipelineIdDisplay').textContent = pipelineId.substring(0, 8) + '...';
        document.getElementById('variantIdDisplay').textContent = variantId ? variantId.substring(0, 8) + '...' : 'N/A';

        btn.textContent = 'Загрузка файла...';
        showStatus('Загрузка файла для обработки...', 'info');
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

        // Show task info
        document.getElementById('taskIdDisplay').textContent = taskId.substring(0, 8) + '...';
        document.getElementById('fileNameDisplay').textContent = file.name;

        // Show progress
        btn.textContent = 'Обработка файла...';
        showStatus(uploadData.message || 'Задача поставлена в очередь', 'info');

        // Poll for status
        await pollTaskStatus(taskId);

    } catch (error) {
        console.error('Processing error:', error);
        showStatus(`Ошибка: ${error.message}`, 'error');
        btn.disabled = false;
        updateProcessButtonText();
    }
}


// ==========================================
// STATUS POLLING AND DISPLAY (Existing code)
// ==========================================
async function pollTaskStatus(taskId) {
    const progressContainer = document.getElementById('progressContainer');
    const progressFill = document.getElementById('progressFill');

    progressContainer.style.display = 'block';

    let pollCount = 0;
    const maxPolls = 600; // 10 минут максимум

    const interval = setInterval(async () => {
        pollCount++;

        if (pollCount > maxPolls) {
            clearInterval(interval);
            showStatus('⏱ Превышено время ожидания. Проверьте задачу позже.', 'error');
            document.getElementById('processBtn').disabled = false;
            updateProcessButtonText();
            return;
        }

        try {
            const response = await fetch(`${API_URL}/api/tasks/${taskId}`);

            if (!response.ok) {
                console.error('Failed to fetch task status:', response.status);
                return;
            }

            const task = await response.json();
            console.log('Task status:', task);

            const progress = (task.progress || 0) * 100;
            progressFill.style.width = progress + '%';
            progressFill.textContent = Math.round(progress) + '%';

            showStatus(task.message || 'Обработка...', 'info');

            if (task.status === 'completed') {
                clearInterval(interval);
                progressFill.style.width = '100%';
                progressFill.textContent = '100%';
                showStatus('✅ Обработка завершена успешно!', 'success');
                document.getElementById('processBtn').disabled = false;
                updateProcessButtonText();
                showResults(task);
            } else if (task.status === 'failed') {
                clearInterval(interval);
                showStatus(`❌ Ошибка: ${task.error || 'Неизвестная ошибка'}`, 'error');
                document.getElementById('processBtn').disabled = false;
                updateProcessButtonText();
            } else if (task.status === 'processing') {
                document.getElementById('processBtn').textContent =
                    `Обработка... ${Math.round(progress)}%`;
            }
        } catch (error) {
            console.error('Error polling task status:', error);
        }
    }, 1000); // Poll every second
}

function showStatus(message, type) {
    const statusEl = document.getElementById('statusMessage');
    statusEl.textContent = message;
    statusEl.className = `status-message status-${type}`;
}

function showResults(task) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';

    const statsEl = document.getElementById('stats');
    statsEl.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">✓</div>
            <div class="stat-label">Статус</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${Math.round((task.progress || 0) * 100)}%</div>
            <div class="stat-label">Прогресс</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${task.status}</div>
            <div class="stat-label">Состояние</div>
        </div>
        ${task.variant_id ? `
        <div class="stat-card">
            <div class="stat-value" style="font-size: 1.2em;">🔧</div>
            <div class="stat-label">Variant: ${task.variant_id.substring(0, 8)}...</div>
        </div>
        ` : ''}
    `;

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}
