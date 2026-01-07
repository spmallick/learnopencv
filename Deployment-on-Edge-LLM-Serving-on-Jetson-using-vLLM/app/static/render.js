// Rendering utilities: bubbles, markdown, code copy buttons, stats
(function(){
  let autoScroll = true;
  function isAtBottom(el, threshold = 4) {
    try { return (el.scrollTop + el.clientHeight) >= (el.scrollHeight - threshold); } catch { return true; }
  }
  (function initScrollTracking(){
    const el = document.getElementById("messages");
    if (!el) return;
    autoScroll = isAtBottom(el);
    el.addEventListener('scroll', () => { autoScroll = isAtBottom(el); });
  })();
  function createAssistantBubble(initialText = "") {
    const messagesDiv = document.getElementById("messages");
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

  function addMessage(text, sender) {
    const messagesDiv = document.getElementById("messages");
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

  function attachCodeCopyButtons(container) {
    const blocks = container.querySelectorAll('pre');
    blocks.forEach(pre => {
      pre.style.position = 'relative';
      if (pre.querySelector('.code-copy-btn')) return;
      const btn = document.createElement('button');
      btn.className = 'code-copy-btn';
      btn.textContent = 'Copy';
      btn.onclick = async () => {
        const codeEl = pre.querySelector('code');
        const text = (codeEl?.innerText || codeEl?.textContent || pre.innerText || '').trim();
        const markError = () => { btn.textContent = 'Error'; setTimeout(() => (btn.textContent = 'Copy'), 1200); };
        const markCopied = () => { btn.textContent = 'Copied'; setTimeout(() => (btn.textContent = 'Copy'), 1200); };
        if (!text) { markError(); return; }
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
            markCopied();
            return;
          }
        } catch (_) {}
        try {
          const ta = document.createElement('textarea');
          ta.value = text;
          ta.style.position = 'fixed';
          ta.style.top = '-9999px';
          ta.setAttribute('readonly', '');
          document.body.appendChild(ta);
          ta.select();
          const ok = document.execCommand('copy');
          document.body.removeChild(ta);
          if (ok) { markCopied(); } else { markError(); }
        } catch (_) {
          markError();
        }
      };
      pre.appendChild(btn);
    });
  }

  function renderMarkdownInto(element, rawText) {
    const messagesDiv = document.getElementById("messages");
    const pinned = autoScroll;
    const html = DOMPurify.sanitize(marked.parse(rawText || ""));
    element.innerHTML = html;
    attachCodeCopyButtons(element);
    if (pinned) messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  function updateStatsFromResponse(data) {
    const statsDiv = document.getElementById("stats");
    try {
      const ct = data.completion_tokens || null;
      const pt = data.prompt_tokens || null;
      const tt = data.total_tokens || null;
      const tps = data.tokens_per_sec || null;
      const dur = data.duration_ms || null;
      const parts = [];
      if (tps) parts.push(`throughput: ${tps.toFixed(2)} tok/s`);
      if (ct) parts.push(`completion: ${ct}`);
      if (pt) parts.push(`prompt: ${pt}`);
      if (tt) parts.push(`total: ${tt}`);
      if (dur) parts.push(`time: ${dur} ms`);
      statsDiv.textContent = parts.join(' • ');
    } catch {
      statsDiv.textContent = '';
    }
  }

  function updateStatsFromStream(footerText, startedAt, contentText) {
    const statsDiv = document.getElementById("stats");
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

  function showImagePreviews(attachments) {
    const messagesDiv = document.getElementById("messages");
    const pinned = autoScroll;
    (attachments || []).filter(a => (a.mime_type || "").startsWith("image/")).forEach(a => {
      const bubble = createAssistantBubble("");
      const img = document.createElement('img');
      img.src = a.url;
      img.alt = a.filename || 'image';
      bubble.appendChild(img);
    });
    if (pinned) messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  window.Render = {
    createAssistantBubble,
    addMessage,
    renderMarkdownInto,
    updateStatsFromResponse,
    updateStatsFromStream,
    showImagePreviews,
  };
})();
