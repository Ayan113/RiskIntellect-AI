/* ============================================================
   RiskIntellect-AI — Application Logic
   Connects the premium dashboard to the FastAPI backend
   ============================================================ */

const API_BASE = '';  // Same origin

// ── Feature Names ──
const FEATURE_NAMES = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'Amount', 'Time'
];

// ── Sample Transactions ──
const SAMPLE_TRANSACTIONS = {
    normal: {
        V1: -1.35, V2: -0.07, V3: 2.53, V4: 1.37, V5: -0.33,
        V6: 0.46, V7: 0.24, V8: 0.10, V9: -0.26, V10: -0.17,
        V11: 1.61, V12: 1.07, V13: 0.49, V14: -0.14, V15: 0.63,
        V16: 0.46, V17: -0.11, V18: -0.58, V19: -0.47, V20: 0.08,
        V21: -0.39, V22: -0.05, V23: -0.11, V24: -0.46, V25: 0.06,
        V26: -0.26, V27: 0.10, V28: -0.19,
        Amount: 149.62, Time: 0,
    },
    suspicious: {
        V1: -3.04, V2: -3.15, V3: 1.09, V4: 2.29, V5: -2.54,
        V6: -1.01, V7: -2.42, V8: 1.10, V9: -3.58, V10: -5.72,
        V11: 3.24, V12: -5.78, V13: -0.56, V14: -9.39, V15: -0.17,
        V16: -4.19, V17: -6.35, V18: -1.94, V19: 1.37, V20: 0.51,
        V21: 0.23, V22: 0.83, V23: -0.04, V24: -0.24, V25: 0.28,
        V26: 0.53, V27: -0.15, V28: -0.33,
        Amount: 14999.00, Time: 7200,
    }
};

let currentRagTab = 'search';
let txnCounter = 1;

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    buildFeatureGrid('feature-grid');
    buildFeatureGrid('copilot-feature-grid');
    refreshHealth();
});

// ═══════ NAVIGATION ═══════
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    document.getElementById(`page-${pageId}`).classList.add('active');
    document.getElementById(`nav-${pageId}`).classList.add('active');
}

// ═══════ FEATURE GRID BUILDER ═══════
function buildFeatureGrid(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = FEATURE_NAMES.map(name => `
    <div class="feature-input-group">
      <label>${name}</label>
      <input type="number" step="any" id="${containerId}-${name}" placeholder="0.0" />
    </div>
  `).join('');
}

function getFeatures(gridId) {
    const features = {};
    FEATURE_NAMES.forEach(name => {
        const el = document.getElementById(`${gridId}-${name}`);
        if (el && el.value !== '') {
            features[name] = parseFloat(el.value);
        }
    });
    return features;
}

function setFeatures(gridId, values) {
    FEATURE_NAMES.forEach(name => {
        const el = document.getElementById(`${gridId}-${name}`);
        if (el && values[name] !== undefined) {
            el.value = values[name];
        }
    });
}

function clearFeatures() {
    FEATURE_NAMES.forEach(name => {
        const el1 = document.getElementById(`feature-grid-${name}`);
        const el2 = document.getElementById(`copilot-feature-grid-${name}`);
        if (el1) el1.value = '';
        if (el2) el2.value = '';
    });
    document.getElementById('scorer-results').style.display = 'none';
}

function loadSampleTransaction(type) {
    const sample = SAMPLE_TRANSACTIONS[type];
    setFeatures('feature-grid', sample);
    setFeatures('copilot-feature-grid', sample);
}

// ═══════ TRANSACTION SCORER ═══════
async function scoreTransaction() {
    const features = getFeatures('feature-grid');
    if (Object.keys(features).length === 0) {
        alert('Please enter at least one feature value');
        return;
    }

    const btn = document.getElementById('btn-score');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Scoring...';

    try {
        const response = await fetch(`${API_BASE}/api/v1/fraud/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                transaction_id: `TXN_${String(txnCounter++).padStart(4, '0')}`,
                features
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Prediction failed');
        }

        displayScorerResult(data);

    } catch (err) {
        alert(`Error: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '🔍 Score Transaction';
    }
}

function displayScorerResult(data) {
    const panel = document.getElementById('scorer-results');
    panel.style.display = 'block';

    // Risk badge
    const badge = document.getElementById('scorer-risk-badge');
    badge.textContent = data.risk_tier;
    badge.className = `risk-badge ${data.risk_tier.toLowerCase()}`;

    // Probability
    const prob = data.fraud_probability;
    document.getElementById('scorer-prob-text').textContent = `${(prob * 100).toFixed(2)}%`;

    const bar = document.getElementById('scorer-prob-bar');
    bar.style.width = `${prob * 100}%`;
    bar.className = `prob-bar-fill ${getRiskClass(prob)}`;

    // Details
    document.getElementById('scorer-txn-id').textContent = data.transaction_id || '—';
    document.getElementById('scorer-is-fraud').textContent = data.is_fraud ? '✅ Yes' : '❌ No';
    document.getElementById('scorer-is-fraud').style.color = data.is_fraud ? 'var(--accent-red)' : 'var(--accent-green)';
    document.getElementById('scorer-threshold').textContent = data.threshold;

    // JSON
    document.getElementById('scorer-json').textContent = JSON.stringify(data, null, 2);

    // Smooth scroll
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ═══════ RAG SEARCH ═══════
function setRagTab(tab) {
    currentRagTab = tab;
    document.querySelectorAll('#rag-tabs .tab').forEach((t, i) => {
        t.classList.toggle('active', (i === 0 && tab === 'search') || (i === 1 && tab === 'answer'));
    });
    const btn = document.getElementById('btn-rag');
    btn.innerHTML = tab === 'search' ? '🔍 Search' : '💬 Get Answer';
}

async function runRAGQuery() {
    const query = document.getElementById('rag-query').value.trim();
    if (!query) {
        alert('Please enter a query');
        return;
    }

    const btn = document.getElementById('btn-rag');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Searching...';

    const endpoint = currentRagTab === 'search' ? '/api/v1/rag/search' : '/api/v1/rag/answer';
    const useReranker = document.getElementById('rag-reranker').checked;

    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 5, use_reranker: useReranker })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Search failed');
        }

        displayRAGResults(data);

    } catch (err) {
        alert(`Error: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = currentRagTab === 'search' ? '🔍 Search' : '💬 Get Answer';
    }
}

function displayRAGResults(data) {
    const panel = document.getElementById('rag-results');
    panel.style.display = 'block';

    // Answer panel (Q&A mode)
    const answerPanel = document.getElementById('rag-answer-panel');
    if (data.answer) {
        answerPanel.style.display = 'block';
        document.getElementById('rag-answer-text').textContent = data.answer;

        const groundedBadge = document.getElementById('rag-grounded-badge');
        groundedBadge.textContent = data.grounded ? 'Grounded ✓' : 'Unverified ⚠';
        groundedBadge.className = `risk-badge ${data.grounded ? 'low' : 'medium'}`;
    } else {
        answerPanel.style.display = 'none';
    }

    // Sources
    const sources = data.results || data.sources || [];
    document.getElementById('rag-result-count').textContent = `${sources.length} results`;

    const sourcesList = document.getElementById('rag-sources-list');
    if (sources.length === 0) {
        sourcesList.innerHTML = '<div class="empty-state"><div class="icon">📄</div><p>No documents found</p></div>';
    } else {
        sourcesList.innerHTML = sources.map((s, i) => `
      <div class="source-card">
        <div class="source-text">${escapeHtml(s.text)}</div>
        <div class="source-meta">
          <span>📁 ${s.source || 'unknown'}</span>
          <span>📊 Score: ${(s.score || 0).toFixed(4)}</span>
          <span>🔗 ${s.retrieval_method || 'hybrid'}</span>
        </div>
      </div>
    `).join('');
    }

    // JSON
    document.getElementById('rag-json').textContent = JSON.stringify(data, null, 2);

    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ═══════ COPILOT ASSESSMENT ═══════
async function runCopilotAssessment() {
    const features = getFeatures('copilot-feature-grid');
    if (Object.keys(features).length === 0) {
        alert('Please enter at least one feature value');
        return;
    }

    const btn = document.getElementById('btn-copilot');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Running Pipeline...';

    const regulatoryQuery = document.getElementById('copilot-query').value.trim() || null;
    const includeExplanation = document.getElementById('copilot-explanation').checked;
    const includeRegulatory = document.getElementById('copilot-regulatory').checked;

    try {
        const response = await fetch(`${API_BASE}/api/v1/copilot/assess`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                transaction_id: `CPT_${String(txnCounter++).padStart(4, '0')}`,
                features,
                regulatory_query: regulatoryQuery,
                include_explanation: includeExplanation,
                include_regulatory_context: includeRegulatory,
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Assessment failed');
        }

        displayCopilotResult(data);

    } catch (err) {
        alert(`Error: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '🤖 Run Full Assessment';
    }
}

function displayCopilotResult(data) {
    const panel = document.getElementById('copilot-results');
    panel.style.display = 'block';

    // Risk badge
    const badge = document.getElementById('copilot-risk-badge');
    badge.textContent = data.risk_tier;
    badge.className = `risk-badge ${data.risk_tier.toLowerCase()}`;

    // Probability bar
    const prob = data.fraud_probability;
    document.getElementById('copilot-prob-text').textContent = `${(prob * 100).toFixed(2)}%`;
    const bar = document.getElementById('copilot-prob-bar');
    bar.style.width = `${prob * 100}%`;
    bar.className = `prob-bar-fill ${getRiskClass(prob)}`;

    // Assessment text
    const assessment = data.risk_assessment || {};
    let assessmentHtml = '';

    if (assessment.explanation) {
        assessmentHtml += `<p><strong style="color: var(--text-accent);">Explanation:</strong> ${escapeHtml(assessment.explanation)}</p>`;
    }
    if (assessment.risk_level) {
        assessmentHtml += `<p><strong style="color: var(--text-accent);">Risk Level:</strong> ${escapeHtml(assessment.risk_level)}</p>`;
    }
    if (assessment.regulatory_basis) {
        assessmentHtml += `<p style="margin-top:8px;"><strong style="color: var(--text-accent);">Regulatory Basis:</strong> ${escapeHtml(assessment.regulatory_basis)}</p>`;
    }
    if (assessment.recommended_action) {
        assessmentHtml += `<p style="margin-top:8px;"><strong style="color: var(--accent-amber);">Recommended Action:</strong> ${escapeHtml(assessment.recommended_action)}</p>`;
    }

    document.getElementById('copilot-assessment-text').innerHTML = assessmentHtml || 'No assessment available';

    // Pipeline metadata
    document.getElementById('copilot-pipeline-json').textContent =
        JSON.stringify(data.pipeline_metadata, null, 2);

    // Full JSON
    document.getElementById('copilot-json').textContent = JSON.stringify(data, null, 2);

    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ═══════ HEALTH CHECK ═══════
async function refreshHealth() {
    try {
        // Root endpoint
        const rootRes = await fetch(`${API_BASE}/`);
        const rootData = await rootRes.json();
        document.getElementById('dash-api-status').textContent = rootRes.ok ? 'Online' : 'Error';
        document.getElementById('dash-api-status').style.color = rootRes.ok ? 'var(--accent-green)' : 'var(--accent-red)';

        // Health endpoint
        const healthRes = await fetch(`${API_BASE}/health`);
        const healthData = await healthRes.json();

        // Update dashboard stats
        const components = healthData.components || {};
        updateDashStat('dash-ml-status', components.ml_engine);
        updateDashStat('dash-rag-status', components.rag_vector_store || components.rag_engine);
        updateDashStat('dash-llm-status', components.llm_layer);

        // Update sidebar status
        const statusDot = document.querySelector('.sidebar-status .status-dot');
        const statusText = document.querySelector('.sidebar-status span:last-child');
        if (healthData.status === 'healthy') {
            statusDot.className = 'status-dot green';
            statusText.textContent = 'All Systems Healthy';
        } else {
            statusDot.className = 'status-dot amber';
            statusText.textContent = 'System Degraded';
        }

        // Health page grid
        const healthGrid = document.getElementById('health-grid');
        healthGrid.innerHTML = Object.entries(components).map(([name, status]) => `
      <div class="health-item">
        <div class="status-indicator ${getStatusClass(status)}"></div>
        <div>
          <div style="font-size: 13px; font-weight: 600;">${formatComponentName(name)}</div>
          <div style="font-size: 11px; color: var(--text-muted);">${status}</div>
        </div>
      </div>
    `).join('');

        // Health JSON
        document.getElementById('health-json').textContent = JSON.stringify(healthData, null, 2);

    } catch (err) {
        document.getElementById('dash-api-status').textContent = 'Offline';
        document.getElementById('dash-api-status').style.color = 'var(--accent-red)';

        const statusDot = document.querySelector('.sidebar-status .status-dot');
        statusDot.className = 'status-dot red';
        document.querySelector('.sidebar-status span:last-child').textContent = 'API Unreachable';
    }
}

// ═══════ UTILITIES ═══════
function getRiskClass(prob) {
    if (prob >= 0.9) return 'critical';
    if (prob >= 0.7) return 'high';
    if (prob >= 0.3) return 'medium';
    return 'low';
}

function updateDashStat(elId, status) {
    const el = document.getElementById(elId);
    if (!el) return;

    const cleanStatus = (status || 'N/A').replace(/_/g, ' ');
    el.textContent = cleanStatus.charAt(0).toUpperCase() + cleanStatus.slice(1);

    if (status === 'ready' || status === 'initialized') {
        el.style.color = 'var(--accent-green)';
    } else if (status && !status.startsWith('error')) {
        el.style.color = 'var(--accent-amber)';
    } else {
        el.style.color = 'var(--accent-red)';
    }
}

function getStatusClass(status) {
    if (status === 'ready' || status === 'initialized') return 'ready';
    if (status && status.startsWith('error')) return 'error';
    return 'degraded';
}

function formatComponentName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
