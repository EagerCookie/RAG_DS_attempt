let currentPipeline = null;
let lastResponseData = null;
let lastQuestion = null;

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

    } catch (error) {
        console.error('Error loading pipeline info:', error);
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
    const evalArea = document.getElementById('evalArea');

    button.disabled = true;
    button.textContent = 'Обработка...';
    evalArea.style.display = 'none'; // Hide previous eval results

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

        if (provider === 'custom' && customModelId) {
            model = customModelId;
        }

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

        // Store for evaluation
        lastResponseData = data;
        lastQuestion = query;

        displayResponse(data, query);

        // Show eval area ready for action
        evalArea.style.display = 'block';
        document.getElementById('evalResults').innerHTML = ''; // Clear old results

    } catch (error) {
        console.error('Error:', error);
        responseArea.innerHTML = `
            <div style="padding: 20px; color: #721c24; background: #f8d7da; border-radius: 8px;">
                <strong>❌ Ошибка:</strong> ${error.message}
            </div>
        `;
    } finally {
        button.disabled = false;
        button.textContent = 'Получить ответ и контекст';
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

    if (data.sources && data.sources.length > 0) {
        html += `
            <div class="sources">
                <strong style="font-size: 1.05em;">📚 Источники (${data.sources.length}):</strong>
                ${data.sources.map((src, idx) => `
                    <div class="source-item">
                        <strong>Источник ${idx + 1}</strong><br>
                        ${escapeHtml(src.content.substring(0, 200))}...
                    </div>
                `).join('')}
            </div>
        `;
    }
    responseArea.innerHTML = html;
}

function formatResponse(text) {
    return escapeHtml(text)
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^(.+)$/, '<p>$1</p>');
}

// EVALUATION LOGIC
async function evaluateCurrentResponse() {
    if (!lastResponseData || !lastQuestion) {
        alert("Нет данных для оценки. Сначала задайте вопрос.");
        return;
    }

    const contexts = lastResponseData.sources.map(s => s.content);

    // Gather metrics
    const metrics = [];
    if (document.getElementById('metric_faithfulness').checked) metrics.push('faithfulness');
    if (document.getElementById('metric_answer_relevancy').checked) metrics.push('answer_relevancy');
    if (document.getElementById('metric_context_precision').checked) metrics.push('context_precision');
    if (document.getElementById('metric_context_recall').checked) metrics.push('context_recall');

    if (metrics.length === 0) {
        alert("Выберите хотя бы одну метрику для оценки!");
        return;
    }

    const btn = document.getElementById('evalBtn');
    const resultsDiv = document.getElementById('evalResults');

    btn.disabled = true;
    const oldText = btn.textContent;
    btn.textContent = "⏳ Вычисляем метрики...";
    resultsDiv.innerHTML = '<div class="eval-loading">Оценка может занять некоторое время (зависит от LLM)...</div>';

    try {
        const response = await fetch(`${API_URL}/api/rag/evaluate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: lastQuestion,
                answer: lastResponseData.answer,
                contexts: contexts,
                metrics: metrics
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Evaluation failed');
        }

        const data = await response.json();
        displayEvalResults(data.scores);

    } catch (e) {
        console.error("Eval error:", e);
        resultsDiv.innerHTML = `<div class="status-error">Ошибка оценки: ${e.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = oldText;
    }
}

function displayEvalResults(scores) {
    const resultsDiv = document.getElementById('evalResults');
    let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">';

    for (const [metric, score] of Object.entries(scores)) {
        // Round to 4 decimals
        const val = typeof score === 'number' ? score.toFixed(4) : score;

        // Color coding for score
        let color = '#333';
        if (typeof score === 'number') {
            if (score > 0.8) color = '#28a745';
            else if (score > 0.5) color = '#ffc107';
            else color = '#dc3545';
        }

        html += `
            <div class="metric-card">
                <span>${formatMetricName(metric)}</span>
                <span class="metric-score" style="color: ${color}">${val}</span>
            </div>
        `;
    }

    html += '</div>';
    resultsDiv.innerHTML = html;
}

function formatMetricName(name) {
    // faithfulness -> Faithfulness
    return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}
