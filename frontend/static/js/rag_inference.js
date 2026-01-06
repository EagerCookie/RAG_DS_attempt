const API_URL = 'http://localhost:8000';
let currentPipeline = null;

// Model options for different providers
const MODEL_OPTIONS = {
    openai: [
        { value: 'gpt-4o', label: 'GPT-4o' },
        { value: 'gpt-4o-mini', label: 'GPT-4o Mini' },
        { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
        { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' }
    ],
    anthropic: [
        { value: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet' },
        { value: 'claude-3-opus-20240229', label: 'Claude 3 Opus' },
        { value: 'claude-3-haiku-20240307', label: 'Claude 3 Haiku' }
    ],
    deepseek: [
        { value: 'deepseek-chat', label: 'DeepSeek Chat' },
        { value: 'deepseek-coder', label: 'DeepSeek Coder' }
    ],
    custom: [
        { value: 'custom', label: 'Custom Model' }
    ]
};

// Load pipelines on page load
window.onload = async () => {
    await loadPipelines();
};

async function loadPipelines() {
    try {
        const response = await fetch(`${API_URL}/api/pipelines`);
        const pipelines = await response.json();

        const select = document.getElementById('pipelineSelect');
        select.innerHTML = '<option value="">-- Выберите пайплайн --</option>' +
            pipelines.map(p =>
                `<option value="${p.id}">${p.name}</option>`
            ).join('');

    } catch (error) {
        console.error('Error loading pipelines:', error);
        alert('Ошибка загрузки пайплайнов. Убедитесь, что API запущен.');
    }
}

async function loadPipelineInfo() {
    const pipelineId = document.getElementById('pipelineSelect').value;

    if (!pipelineId) {
        document.getElementById('pipelineInfo').style.display = 'none';
        document.getElementById('pipelineStatus').textContent = 'Не выбран';
        document.getElementById('pipelineStatus').className = 'status-badge status-ready';
        currentPipeline = null;
        return;
    }

    try {
        // Load pipeline details
        const pipelineResp = await fetch(`${API_URL}/api/pipelines/${pipelineId}`);
        const pipeline = await pipelineResp.json();

        // Load pipeline files
        const filesResp = await fetch(`${API_URL}/api/pipelines/${pipelineId}/files`);
        const filesData = await filesResp.json();

        currentPipeline = {
            id: pipeline.pipeline_id,
            config: pipeline.config,
            vectorDbId: filesData.vector_db_identifier,
            filesCount: filesData.total_files,
            chunksCount: filesData.files.reduce((sum, f) => sum + (f.chunks_count || 0), 0)
        };

        // Update UI
        document.getElementById('vectorDbId').textContent = currentPipeline.vectorDbId;
        document.getElementById('filesCount').textContent = currentPipeline.filesCount;
        document.getElementById('chunksCount').textContent = currentPipeline.chunksCount;
        document.getElementById('pipelineInfo').style.display = 'block';
        document.getElementById('pipelineStatus').textContent = 'Готов';
        document.getElementById('pipelineStatus').className = 'status-badge status-ready';

    } catch (error) {
        console.error('Error loading pipeline info:', error);
        document.getElementById('pipelineStatus').textContent = 'Ошибка';
        document.getElementById('pipelineStatus').className = 'status-badge status-error';
    }
}

function updateModelOptions() {
    const provider = document.getElementById('llmProvider').value;
    const modelSelect = document.getElementById('llmModel');
    const customUrlDiv = document.getElementById('customApiUrl');

    const options = MODEL_OPTIONS[provider] || MODEL_OPTIONS.openai;
    modelSelect.innerHTML = options.map(opt =>
        `<option value="${opt.value}">${opt.label}</option>`
    ).join('');

    customUrlDiv.style.display = provider === 'custom' ? 'block' : 'none';
}

async function askQuestion() {
    const query = document.getElementById('queryInput').value.trim();

    if (!query) {
        alert('Пожалуйста, введите вопрос');
        return;
    }

    if (!currentPipeline) {
        alert('Пожалуйста, выберите пайплайн');
        return;
    }

    const button = document.getElementById('askButton');
    const responseArea = document.getElementById('responseArea');

    button.disabled = true;
    button.textContent = 'Обработка...';

    // Show loading state
    responseArea.innerHTML = `
        <div style="text-align: center; padding: 40px;">
            <div class="loading"></div>
            <p style="margin-top: 20px; color: #666;">Поиск в базе знаний и генерация ответа...</p>
        </div>
    `;

    try {
        const provider = document.getElementById('llmProvider').value;
        let model = document.getElementById('llmModel').value;
        const topK = parseInt(document.getElementById('topK').value);
        const temperature = parseFloat(document.getElementById('temperature').value);
        const customUrl = document.getElementById('customUrl').value;
        const customModelId = document.getElementById('customModelId').value;

        // Для custom провайдера используем customModelId если он указан
        if (provider === 'custom' && customModelId) {
            model = customModelId;
        }

        // Call RAG endpoint
        const response = await fetch(`${API_URL}/api/rag/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pipeline_id: currentPipeline.id,
                query: query,
                llm_provider: provider,
                llm_model: model,
                top_k: topK,
                temperature: temperature,
                custom_api_url: customUrl || null
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Ошибка при выполнении запроса');
        }

        const data = await response.json();

        displayResponse(data, query);

    } catch (error) {
        console.error('Error:', error);
        responseArea.innerHTML = `
            <div style="padding: 20px; color: #721c24; background: #f8d7da; border-radius: 8px;">
                <strong>❌ Ошибка:</strong> ${error.message}
            </div>
        `;
    } finally {
        button.disabled = false;
        button.textContent = 'Задать вопрос';
    }
}

function displayResponse(data, query) {
    const responseArea = document.getElementById('responseArea');

    const timestamp = new Date().toLocaleTimeString('ru-RU');

    let html = `
        <div class="response-header">
            <strong style="font-size: 1.1em;">💬 Ответ</strong>
            <span style="color: #999; font-size: 0.9em;">${timestamp}</span>
        </div>
        
        <div style="background: #e3f2fd; padding: 12px; border-radius: 8px; margin-bottom: 15px;">
            <strong>❓ Вопрос:</strong> ${escapeHtml(query)}
        </div>
        
        <div class="response-content">
            ${formatResponse(data.answer)}
        </div>
    `;

    // Add sources if available
    if (data.sources && data.sources.length > 0) {
        html += `
            <div class="sources">
                <strong style="font-size: 1.05em;">📚 Источники (${data.sources.length}):</strong>
                ${data.sources.map((src, idx) => `
                    <div class="source-item">
                        <strong>Источник ${idx + 1}</strong><br>
                        ${escapeHtml(src.content.substring(0, 200))}...
                        ${src.metadata ? `<br><small style="color: #999;">${JSON.stringify(src.metadata)}</small>` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    }

    // Add metadata
    if (data.metadata) {
        html += `
            <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; font-size: 0.9em;">
                <strong>ℹ️ Метаданные:</strong><br>
                <strong>Модель:</strong> ${data.metadata.llm_model || 'N/A'}<br>
                <strong>Время обработки:</strong> ${data.metadata.processing_time || 'N/A'}<br>
                <strong>Найдено документов:</strong> ${data.metadata.retrieved_docs || 'N/A'}
            </div>
        `;
    }

    responseArea.innerHTML = html;
}

function formatResponse(text) {
    // Simple markdown-like formatting
    return escapeHtml(text)
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^(.+)$/, '<p>$1</p>');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Handle Enter key in textarea (Shift+Enter for new line)
document.getElementById('queryInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
});
