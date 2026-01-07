// Network utilities: streaming, models, errors, throughput footer stripping
(function(){
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

  async function safeExtractError(response) {
    try {
      const data = await response.json();
      return data.detail || data.error?.message || response.statusText;
    } catch (_) {
      try { return await response.text(); } catch { return response.statusText; }
    }
  }

  async function loadModels() {
    const modelSelect = document.getElementById("modelSelect");
    const modelCaps = document.getElementById("modelCaps");
    const visionToggle = document.getElementById("visionToggle");
    let models = [];
    let currentModel = null;
    try {
      const res = await fetch('/models');
      const data = await res.json();
      models = data.models || [];
      modelSelect.innerHTML = '';
      for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.id;
        modelSelect.appendChild(opt);
      }
      const defId = data.default || null;
      currentModel = (defId && models.find(x => x.id === defId)?.id) || models[0]?.id || null;
      modelSelect.value = currentModel || '';
      updateModelCaps();
    } catch (e) {
      modelSelect.innerHTML = '<option value="">(failed to load)</option>';
    }
    function updateModelCaps() {
      const item = models.find(m => m.id === currentModel);
      if (!item) { modelCaps.textContent = ''; visionToggle.checked = false; return; }
      // Set toggle to the probed vision capability by default
      visionToggle.checked = item.vision;
      modelCaps.textContent = item.vision ? 'Vision-capable' : 'Text-only';
    }
    modelSelect.onchange = () => {
      currentModel = modelSelect.value || null;
      updateModelCaps();
      const item = models.find(m => m.id === currentModel);
      const fileInput = document.getElementById("fileInput");
      const fileList = document.getElementById("fileList");
      if (item && !item.vision && !visionToggle.checked) {
        if (Array.from(fileInput.files || []).some(f => !(f.type || '').startsWith('text/'))) {
          fileInput.value = '';
          fileList.textContent = '';
        }
      }
    };
    // Handle vision toggle changes
    visionToggle.onchange = () => {
      const item = models.find(m => m.id === currentModel);
      if (!item) return;
      // Update display based on toggle state
      modelCaps.textContent = visionToggle.checked ? 'Vision-capable (override)' : 'Text-only (override)';
      // Re-validate selected files based on new vision state
      const fileInput = document.getElementById("fileInput");
      const fileList = document.getElementById("fileList");
      const files = Array.from(fileInput.files || []);
      if (files.length > 0) {
        // Use Attach.validateSelectionForModel to validate with new vision state
        if (typeof Attach !== 'undefined' && Attach.validateSelectionForModel) {
          const ok = Attach.validateSelectionForModel(files, models, currentModel, visionToggle.checked);
          if (!ok.ok) {
            fileInput.value = '';
            fileList.textContent = '';
          }
        }
      }
    };

    return { models, currentModel, updateModelCaps };
  }

  window.Net = {
    stripThroughputFooter,
    safeExtractError,
    loadModels,
    checkWebSearchStatus: async function() {
      try {
        const res = await fetch('/search/status');
        const data = await res.json();
        return data.available || false;
      } catch {
        return false;
      }
    }
  };
})();
