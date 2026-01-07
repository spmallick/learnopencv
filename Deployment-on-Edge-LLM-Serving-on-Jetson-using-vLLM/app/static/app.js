// App logic extracted from index.html
(function(){
  const messagesDiv = document.getElementById("messages");
  const inputField = document.getElementById("inputText");
  const sendBtn = document.getElementById("sendBtn");
  const resetBtn = document.getElementById("resetBtn");
  const attachBtn = document.getElementById("attachBtn");
  const fileInput = document.getElementById("fileInput");
  const fileList = document.getElementById("fileList");
  const modelSelect = document.getElementById("modelSelect");
  const modelCaps = document.getElementById("modelCaps");
  const visionToggle = document.getElementById("visionToggle");
  const webSearchToggle = document.getElementById("webSearchToggle");
  const ragToggle = document.getElementById("ragToggle");
  const ragDocList = document.getElementById("ragDocList");
  const ragDocCount = document.getElementById("ragDocCount");
  const statsDiv = document.getElementById("stats");

  let userId = "user-" + Math.random().toString(36).substring(2, 8);
  let models = [];
  let currentModel = null;

  let autoScroll = true;
  function updateAutoScroll() {
    autoScroll = (messagesDiv.scrollTop + messagesDiv.clientHeight) >= (messagesDiv.scrollHeight - 4);
  }
  messagesDiv.addEventListener('scroll', updateAutoScroll);
  updateAutoScroll();

  function addMessage(text, sender) {
    const pinned = autoScroll;
    const bubble = document.createElement("div");
    bubble.classList.add("bubble", sender);
    bubble.textContent = text;

    const line = document.createElement("div");
    line.style.display = "flex";
    line.style.justifyContent = sender === "user" ? "flex-end" : "flex-start";
    line.appendChild(bubble);

    messagesDiv.appendChild(line);
    if (pinned) messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  function createAssistantBubble(initialText = "") {
    const pinned = autoScroll;
    const bubble = document.createElement("div");
    bubble.classList.add("bubble", "assistant");
    bubble.innerText = initialText;

    const line = document.createElement("div");
    line.style.display = "flex";
    line.style.justifyContent = "flex-start";
    line.appendChild(bubble);

    messagesDiv.appendChild(line);
    if (pinned) messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return bubble;
  }

  async function sendMessage() {
    // Ensure models are loaded so currentModel is valid
    try { await modelsReady; } catch(_) {}
    const text = inputField.value.trim();
    if (!text) return;
    if (!currentModel) {
      const bubble = createAssistantBubble("");
      renderMarkdownInto(bubble, "Error: No model selected. Please choose a model from the dropdown.");
      return;
    }

    addMessage(text, "user");
    inputField.value = "";

    try {
      await streamAssistantResponse(text);
    } catch (e) {
      console.warn("Stream failed; falling back to /chat", e);
      const attachments = await uploadSelectedFiles();
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, message: text, attachments, model: currentModel, vision_enabled: visionToggle.checked, web_search: webSearchToggle.checked, rag_enabled: ragToggle.checked }),
      });
      if (!response.ok) {
        const err = await safeExtractError(response);
        const bubble = createAssistantBubble("");
        renderMarkdownInto(bubble, `Error: ${err}`);
        return;
      }
      const data = await response.json();
      const bubble = createAssistantBubble("");
      renderMarkdownInto(bubble, data.reply || "");
      updateStatsFromResponse(data);
      showImagePreviews(attachments);
    }
  }

  sendBtn.onclick = sendMessage;
  attachBtn.onclick = () => fileInput.click();
  fileInput.onchange = async () => {
    const files = Array.from(fileInput.files || []);
    const ok = Attach.validateSelectionForModel(files, models, currentModel, visionToggle.checked);
    if (!ok.ok) { fileInput.value = ""; fileList.textContent = ""; return; }
    
    // Check for PDFs and offer to index for RAG
    const pdfFiles = files.filter(f => f.name.toLowerCase().endsWith('.pdf'));
    const otherFiles = files.filter(f => !f.name.toLowerCase().endsWith('.pdf'));
    
    // Index PDFs for RAG automatically
    for (const pdf of pdfFiles) {
      await uploadRagDocument(pdf);
    }
    
    // Show other files in the list
    if (otherFiles.length > 0) {
      const names = otherFiles.map(f => `• ${f.name}`);
      fileList.textContent = names.join("\n");
    } else if (pdfFiles.length > 0) {
      // Clear file input if only PDFs (they're indexed, not attached)
      fileInput.value = "";
    }
  };

  inputField.addEventListener("keypress", function (e) {
    if (e.key === "Enter") sendMessage();
  });

  resetBtn.onclick = async function () {
    try {
      await fetch("/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId })
      });
      messagesDiv.innerHTML = "";
      const bubble = createAssistantBubble("");
      renderMarkdownInto(bubble, "_Chat reset._");
    } catch (err) {
      console.error("Failed to reset chat", err);
    }
  }

  // Configure marked to highlight code blocks
  marked.setOptions({
    highlight: function(code, lang) {
      try {
        if (lang && window.hljs?.getLanguage(lang)) {
          return window.hljs.highlight(code, { language: lang }).value;
        }
        return window.hljs?.highlightAuto(code).value || code;
      } catch { return code; }
    }
  });

  function renderMarkdownInto(element, rawText) {
    Render.renderMarkdownInto(element, rawText);
  }

  function attachCodeCopyButtons(container) { Render.renderMarkdownInto(container, container.innerHTML); }

  function showImagePreviews(attachments) { Render.showImagePreviews(attachments); }

  async function safeExtractError(response) {
    try {
      const data = await response.json();
      return data.detail || data.error?.message || response.statusText;
    } catch (_) {
      try { return await response.text(); } catch { return response.statusText; }
    }
  }

  let modelsReady = null;
  async function loadModels() {
    const info = await Net.loadModels();
    models = info.models || models;
    currentModel = info.currentModel || currentModel;
  }

  // Check web search availability and update toggle
  async function initWebSearch() {
    const available = await Net.checkWebSearchStatus();
    if (webSearchToggle) {
      webSearchToggle.disabled = !available;
      if (!available) {
        webSearchToggle.title = "Web search is not available (API key not configured)";
        webSearchToggle.parentElement.style.opacity = "0.5";
      } else {
        webSearchToggle.title = "Enable web search to augment responses with real-time information";
      }
    }
  }

  // Start loading models immediately
  modelsReady = loadModels();
  
  // Initialize web search status
  initWebSearch();

  function updateModelCaps() {
    const item = models.find(m => m.id === currentModel);
    if (!item) { modelCaps.textContent = ''; visionToggle.checked = false; return; }
    // Set toggle based on model's probed vision capability
    visionToggle.checked = item.vision;
    modelCaps.textContent = item.vision ? 'Vision-capable' : 'Text-only';
  }

  modelSelect.onchange = () => {
    currentModel = modelSelect.value || null;
    updateModelCaps();
    const item = models.find(m => m.id === currentModel);
    if (item && !item.vision) {
      if (Array.from(fileInput.files || []).some(f => !(f.type || '').startsWith('text/'))) {
        fileInput.value = '';
        fileList.textContent = '';
      }
    }
  }

  // models are already loading via modelsReady

  async function uploadSelectedFiles() { return Attach.uploadSelectedFiles(fileInput); }

  async function streamAssistantResponse(promptText) {
    const bubble = createAssistantBubble("...");
    let buffer = "";
    const attachments = await uploadSelectedFiles();
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, message: promptText, attachments, model: currentModel, vision_enabled: visionToggle.checked, web_search: webSearchToggle.checked, rag_enabled: ragToggle.checked }),
    });

    if (!response.ok || !response.body) {
      const err = await Net.safeExtractError(response);
      const eb = createAssistantBubble("");
      renderMarkdownInto(eb, `Error: ${err}`);
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let startedAt = performance.now();
    let latestFooter = null;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;
      const { clean, footer } = Net.stripThroughputFooter(buffer);
      if (footer) latestFooter = footer;
      renderMarkdownInto(bubble, clean);
    }

    const { clean, footer } = Net.stripThroughputFooter(buffer);
    renderMarkdownInto(bubble, clean);
    showImagePreviews(attachments);
    Render.updateStatsFromStream(footer || latestFooter, startedAt, clean);
  }

  function updateStatsFromResponse(data) { Render.updateStatsFromResponse(data); }

  function updateStatsFromStream(footerText, startedAt, contentText) {
    try {
      const now = performance.now();
      const durMs = Math.max(1, now - startedAt);
      const m = footerText && footerText.match(/\[throughput\]\s+duration_ms=(\d+)\s+tokens_per_sec=([\d.]+)\s+approx_tokens=(\d+)/);
      if (m) {
        statsDiv.textContent = `throughput: ${parseFloat(m[2]).toFixed(2)} tok/s • completion: ${m[3]} • time: ${m[1]} ms`;
        return;
      }
      const approxTokens = Math.max(1, (contentText || '').length / 4);
      const tps = approxTokens / (durMs / 1000);
      statsDiv.textContent = `throughput: ${tps.toFixed(2)} tok/s • time: ${Math.round(durMs)} ms`;
    } catch {
      statsDiv.textContent = '';
    }
  }

  function stripThroughputFooter(text) {
    try {
      const src = String(text || '');
      const reLine = /(^|\n)\s*\[throughput\][^\n]*\n?/g;
      let lastFooter = null;
      const matches = src.match(/\[throughput\][^\n]*/g);
      if (matches && matches.length) lastFooter = matches[matches.length - 1].trim();
      const clean = src.replace(reLine, (m, g1) => g1 ? g1 : '');
      return { clean: clean.trim(), footer: lastFooter };
    } catch {
      return { clean: text, footer: null };
    }
  }

  // ============== RAG Document Management ==============
  
  async function loadRagDocuments() {
    try {
      const res = await fetch(`/rag/documents?user_id=${encodeURIComponent(userId)}`);
      const data = await res.json();
      const docs = data.documents || [];
      
      // Update doc count badge
      if (ragDocCount) {
        ragDocCount.textContent = docs.length > 0 ? `(${docs.length} doc${docs.length > 1 ? 's' : ''} indexed)` : '';
      }
      
      if (docs.length === 0) {
        ragDocList.innerHTML = '<em style="color:#888;">No documents indexed</em>';
        return;
      }
      
      ragDocList.innerHTML = docs.map(doc => `
        <div style="display:flex; justify-content:space-between; align-items:center; padding:4px 6px; margin:4px 0; background:#f8f9fb; border-radius:4px; border:1px solid #eee;">
          <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:160px;" title="${doc.filename}">📄 ${doc.filename}</span>
          <button onclick="window.deleteRagDoc('${doc.doc_id}')" style="padding:2px 6px; font-size:10px; background:#dc3545; color:white; border:none; border-radius:3px; cursor:pointer;">✕</button>
        </div>
      `).join('');
    } catch (e) {
      console.error('Failed to load RAG documents:', e);
    }
  }
  
  async function uploadRagDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Show indexing status in file list
    fileList.textContent = `⏳ Indexing ${file.name}...`;
    
    try {
      const res = await fetch(`/rag/upload?user_id=${encodeURIComponent(userId)}`, {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) {
        const err = await res.json();
        fileList.textContent = `❌ Failed: ${err.detail || 'Unknown error'}`;
        return false;
      }
      
      const data = await res.json();
      console.log('Document indexed:', data);
      fileList.textContent = `✅ Indexed: ${file.name} (${data.chunk_count} chunks)`;
      await loadRagDocuments();
      return true;
    } catch (e) {
      console.error('Upload failed:', e);
      fileList.textContent = '❌ Upload failed';
      return false;
    }
  }
  
  window.deleteRagDoc = async function(docId) {
    if (!confirm('Delete this document from RAG index?')) return;
    
    try {
      const res = await fetch(`/rag/documents/${docId}?user_id=${encodeURIComponent(userId)}`, {
        method: 'DELETE',
      });
      
      if (!res.ok) {
        alert('Failed to delete document');
        return;
      }
      
      await loadRagDocuments();
    } catch (e) {
      console.error('Delete failed:', e);
    }
  };
  
  // Initialize RAG status and load documents
  async function initRag() {
    try {
      const res = await fetch('/rag/status');
      const data = await res.json();
      if (ragToggle) {
        ragToggle.disabled = !data.available;
        if (!data.available) {
          ragToggle.title = 'RAG is not available';
          ragToggle.parentElement.style.opacity = '0.5';
        }
      }
      await loadRagDocuments();
    } catch (e) {
      console.error('RAG init failed:', e);
      if (ragToggle) {
        ragToggle.disabled = true;
        ragToggle.parentElement.style.opacity = '0.5';
      }
    }
  }
  
  initRag();
  // ============== End RAG Document Management ==============
})();
