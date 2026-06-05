// Detect API Base URL
const API_BASE = window.location.origin.includes('http') ? window.location.origin : 'http://127.0.0.1:8000';

// App State
let network = null;
let graphDataRaw = { nodes: [], edges: [] };
let isProcessing = false;

// DOM Elements
const apiStatusIndicator = document.getElementById('api-status-indicator');
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const processStatus = document.getElementById('process-status');
const progressBarFill = document.querySelector('.progress-bar-fill');
const processMessage = document.getElementById('process-message');
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');
const chatMessages = document.getElementById('chat-messages');
const btnPrompts = document.querySelectorAll('.btn-prompt');
const nodeInfoPanel = document.getElementById('node-info-panel');
const infoNodeType = document.getElementById('info-node-type');
const infoNodePage = document.getElementById('info-node-page');
const infoNodeTitle = document.getElementById('info-node-title');
const infoNodeText = document.getElementById('info-node-text');
const contextSourcesPanel = document.getElementById('context-sources-panel');
const contextSourcesList = document.getElementById('context-sources-list');

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
    checkBackendStatus();
    setupUploadHandlers();
    setupChatHandlers();
    
    // Bind Load Graph from DB button
    const loadDbBtn = document.getElementById('load-db-btn');
    if (loadDbBtn) {
        loadDbBtn.addEventListener('click', () => {
            loadAndRenderGraph();
        });
    }
});

// 1. Check Server Connection & Settings
async function checkBackendStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        if (!response.ok) throw new Error("Status check failed");
        
        const data = await response.json();
        const dot = apiStatusIndicator.querySelector('.status-dot');
        const label = apiStatusIndicator.querySelector('.status-label');

        if (data.gemini_api_key_configured) {
            dot.className = "status-dot active";
            label.textContent = "Gemini AI Online";
        } else {
            dot.className = "status-dot warning";
            label.textContent = "Gemini Key Missing (.env)";
            appendMessage('assistant', `⚠️ <strong>Gemini API Key is missing.</strong> Please add your API key in the <code>backend/.env</code> file as: <br><code>GEMINI_API_KEY=your_key_here</code> to enable AI processing. Read the <strong>README.md</strong> at the root for a step-by-step guide.`);
        }

        if (data.is_indexed) {
            enableQueryInterface();
            loadAndRenderGraph();
        } else {
            appendMessage('assistant', `💡 <strong>No documents loaded.</strong> Drag and drop a T&C PDF file on the panel above to generate the legal knowledge graph.`);
        }
    } catch (error) {
        console.error("Error checking backend status:", error);
        const dot = apiStatusIndicator.querySelector('.status-dot');
        const label = apiStatusIndicator.querySelector('.status-label');
        dot.className = "status-dot error";
        label.textContent = "Server Offline";
        appendMessage('assistant', `❌ <strong>Cannot connect to Python Backend.</strong> Please make sure you have installed the requirements and started the server by running:<br><code>python app.py</code> inside the <code>backend/</code> folder.`);
    }
}

// Enable Inputs once documents are ready
function enableQueryInterface() {
    queryInput.disabled = false;
    sendBtn.disabled = false;
    btnPrompts.forEach(btn => btn.disabled = false);
}

// 2. File Upload & Ingest Controls
function setupUploadHandlers() {
    // Click drop-zone to open input
    dropZone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            uploadFile(fileInput.files[0]);
        }
    });

    // Drag-and-drop animations
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            uploadFile(files[0]);
        }
    });
}

// API Call to upload and process PDF
async function uploadFile(file) {
    if (isProcessing) return;
    
    isProcessing = true;
    processStatus.style.display = 'block';
    progressBarFill.style.width = '5%';
    processMessage.textContent = "Uploading PDF document...";
    
    // Animate fake progress up to 90%
    let progress = 5;
    const progressInterval = setInterval(() => {
        if (progress < 90) {
            progress += Math.floor(Math.random() * 8) + 2;
            progressBarFill.style.width = `${Math.min(progress, 90)}%`;
            
            if (progress > 30 && progress < 60) {
                processMessage.textContent = "Parsing PDF & splitting clauses...";
            } else if (progress >= 60) {
                processMessage.textContent = "Extracting legal relationships with Gemini...";
            }
        }
    }, 450);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Ingestion failed");
        }

        const data = await response.json();
        
        // Progress Complete
        clearInterval(progressInterval);
        progressBarFill.style.width = '100%';
        processMessage.textContent = "Indexing complete!";
        
        setTimeout(() => {
            processStatus.style.display = 'none';
        }, 1500);

        appendMessage('assistant', `✅ <strong>Success!</strong> "${file.name}" has been successfully parsed and mapped into the knowledge graph (${data.chunks_count} sections indexed). You can now ask questions about this document.`);
        
        enableQueryInterface();
        loadAndRenderGraph();
        
    } catch (error) {
        clearInterval(progressInterval);
        progressBarFill.style.width = '0%';
        processMessage.textContent = "Processing failed.";
        appendMessage('assistant', `❌ <strong>Upload Failed:</strong> ${error.message}`);
        console.error(error);
    } finally {
        isProcessing = false;
        fileInput.value = ''; // Reset input
    }
}

// 3. Render Vis.js Graph Representation
async function loadAndRenderGraph() {
    const container = document.getElementById('graph-network');
    
    try {
        const response = await fetch(`${API_BASE}/api/graph`);
        if (!response.ok) throw new Error("Could not load graph");
        
        graphDataRaw = await response.json();
        
        if (graphDataRaw.nodes.length === 0) {
            container.innerHTML = `
                <div class="graph-placeholder">
                    <div class="placeholder-icon">🕸️</div>
                    <h3>No Knowledge Graph Ingested</h3>
                    <p>Document is empty or not parsed. Upload a PDF above.</p>
                </div>
            `;
            return;
        }

        // Configure styles depending on Node Entity type
        const styledNodes = graphDataRaw.nodes.map(node => {
            let color = '#3b82f6'; // Default Slate Blue for Clause
            let shape = 'dot';
            let fontColor = '#f3f4f6';
            let size = 16;
            
            if (node.type === 'Actor') {
                color = '#10b981'; // Emerald Green
                shape = 'box';
                fontColor = '#07080b';
                size = 18;
            } else if (node.type === 'Definition') {
                color = '#8b5cf6'; // Orchid Purple
                shape = 'diamond';
                size = 14;
            } else if (node.type === 'Risk') {
                color = '#ef4444'; // Crimson Red
                shape = 'dot';
                size = 20;
            }

            return {
                id: node.id,
                label: node.label,
                title: node.type, // Tooltip
                shape: shape,
                size: size,
                color: {
                    background: color,
                    border: 'rgba(255, 255, 255, 0.08)',
                    highlight: {
                        background: '#3b82f6',
                        border: '#f3f4f6'
                    }
                },
                font: {
                    color: fontColor,
                    size: 11,
                    face: 'Plus Jakarta Sans',
                    multi: true
                },
                widthConstraint: {
                    maximum: 140
                },
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.4)',
                    size: 8,
                    x: 0,
                    y: 3
                },
                // Store additional raw properties
                rawText: node.text,
                rawPage: node.page,
                rawType: node.type
            };
        });

        // Configure Edges
        const styledEdges = graphDataRaw.edges.map(edge => {
            return {
                from: edge.from,
                to: edge.to,
                label: edge.label,
                arrows: 'to',
                color: {
                    color: 'rgba(255, 255, 255, 0.08)',
                    highlight: '#3b82f6',
                    hover: '#3b82f6'
                },
                font: {
                    color: '#9ca3af',
                    size: 9,
                    face: 'Plus Jakarta Sans',
                    strokeWidth: 0 // Remove white shadow around text
                },
                smooth: {
                    type: 'curvedCW',
                    roundness: 0.15
                }
            };
        });

        const data = {
            nodes: new vis.DataSet(styledNodes),
            edges: new vis.DataSet(styledEdges)
        };

        const options = {
            physics: {
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {
                    gravitationalConstant: -120, // Stronger repulsion to prevent clustering
                    centralGravity: 0.005,
                    springLength: 160,          // Longer spring distance
                    springConstant: 0.04,
                    nodeDistance: 180
                },
                maxVelocity: 45,
                minVelocity: 0.1,
                stabilization: {
                    iterations: 200,
                    updateInterval: 25
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 200
            }
        };

        // Initialize Vis Network
        container.innerHTML = ""; // Clear placeholders
        network = new vis.Network(container, data, options);

        // Bind Clicks to Node Detail Side Card
        network.on("selectNode", (params) => {
            const selectedNodeId = params.nodes[0];
            const nodeData = styledNodes.find(n => n.id === selectedNodeId);
            if (nodeData) {
                displayNodeDetails(nodeData);
            }
        });

        network.on("deselectNode", () => {
            closeNodeInfo();
        });

    } catch (e) {
        console.error("Error rendering graph:", e);
        container.innerHTML = `<div style="padding: 40px; color: var(--accent-crimson);">Error loading graph network structure.</div>`;
    }
}

// Show info panel
function displayNodeDetails(node) {
    infoNodeType.textContent = node.rawType;
    // Set color based on node type
    infoNodeType.style.background = node.color.background;
    infoNodeType.style.color = node.rawType === 'Actor' ? '#000' : '#fff';
    
    if (node.rawType === 'Clause' && node.rawPage) {
        infoNodePage.style.display = 'inline-block';
        infoNodePage.textContent = `Page ${node.rawPage}`;
    } else {
        infoNodePage.style.display = 'none';
    }

    infoNodeTitle.textContent = node.label;
    infoNodeText.innerHTML = node.rawText ? node.rawText.replace(/\n/g, '<br>') : "No content text associated with this entity.";
    nodeInfoPanel.style.display = 'block';
}

function closeNodeInfo() {
    nodeInfoPanel.style.display = 'none';
    if (network) network.unselectNodes();
}

// 4. Chat Q&A Interaction
function setupChatHandlers() {
    // Submit on click
    sendBtn.addEventListener('click', () => handleQuerySubmit());

    // Submit on Enter keypress
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleQuerySubmit();
        }
    });
}

// Handle trigger from Quick Suggestion buttons
function sendQuickPrompt(promptText) {
    queryInput.value = promptText;
    handleQuerySubmit();
}

// Process query submit
async function handleQuerySubmit() {
    const query = queryInput.value.trim();
    if (!query || isProcessing) return;

    // Append user message
    appendMessage('user', query);
    queryInput.value = ''; // Reset input
    
    // Add loading bubble
    const loadingId = appendMessage('assistant', `
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `);

    try {
        const response = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        if (!response.ok) throw new Error("Failed to search query");
        const data = await response.json();

        // Remove loading bubble
        removeMessage(loadingId);

        // Format Gemini Response Markdown (Basic markdown parsing)
        let formattedAnswer = formatMarkdownText(data.answer);
        appendMessage('assistant', formattedAnswer);

        // Map and Highlight Hybrid Context references
        if (data.subgraph && data.subgraph.nodes.length > 0) {
            highlightRAGTraversalPath(data.subgraph);
        } else {
            contextSourcesPanel.style.display = 'none';
        }

    } catch (error) {
        removeMessage(loadingId);
        appendMessage('assistant', `❌ <strong>Query Error:</strong> ${error.message}`);
        console.error(error);
    }
}

// Helper to format basic markdown responses from LLM
function formatMarkdownText(text) {
    let clean = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
        
    // Bold matches
    clean = clean.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Code block matches
    clean = clean.replace(/`(.*?)`/g, '<code>$1</code>');
    // Newlines / list matches
    clean = clean.replace(/^\s*-\s*(.*?)$/gm, '<li>$1</li>');
    clean = clean.replace(/(<li>.*?<\/li>)/g, '<ul>$1</ul>');
    // Fix double nested lists
    clean = clean.replace(/<\/ul>\s*<ul>/g, '');
    // Line breaks
    clean = clean.replace(/\n/g, '<br>');
    return clean;
}

// Helper to insert bubble to chat stream
function appendMessage(sender, text) {
    const msgId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5);
    const bubble = document.createElement('div');
    bubble.className = `chat-message ${sender}`;
    bubble.id = msgId;
    bubble.innerHTML = text;
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msgId;
}

function removeMessage(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

// 5. Highlight Context Traversal Path on Graph Visualizer
function highlightRAGTraversalPath(subgraph) {
    if (!network) return;

    // Show source context panel
    contextSourcesPanel.style.display = 'block';
    contextSourcesList.innerHTML = '';

    const highlightedNodeIds = subgraph.nodes.map(n => n.id);
    
    // Update chip items below graph
    subgraph.nodes.forEach(node => {
        const chip = document.createElement('div');
        chip.className = 'context-chip';
        chip.innerHTML = `<span class="context-chip-icon">📍</span> ${node.label}`;
        chip.addEventListener('click', () => {
            // Focus camera on node on click
            network.focus(node.id, {
                scale: 1.2,
                animation: { duration: 800, easingFunction: 'easeInOutQuad' }
            });
            network.selectNodes([node.id]);
            // Retrieve node properties from graph database raw data to display details
            const rawNode = graphDataRaw.nodes.find(n => n.id === node.id);
            if (rawNode) {
                displayNodeDetails({
                    label: rawNode.label,
                    color: { background: rawNode.type === 'Clause' ? '#3b82f6' : (rawNode.type === 'Actor' ? '#10b981' : (rawNode.type === 'Definition' ? '#8b5cf6' : '#ef4444')) },
                    rawType: rawNode.type,
                    rawPage: rawNode.page,
                    rawText: rawNode.text
                });
            }
        });
        contextSourcesList.appendChild(chip);
    });

    // Visually pulse/select nodes inside network
    network.selectNodes(highlightedNodeIds);
    
    // Fits view dynamically to encapsulate these highlighted nodes
    network.fit({
        nodes: highlightedNodeIds,
        animation: { duration: 1000, easingFunction: 'easeInOutQuad' }
    });
}
