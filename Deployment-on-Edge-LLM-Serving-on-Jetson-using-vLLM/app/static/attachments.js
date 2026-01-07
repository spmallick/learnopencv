// Attachment handling: selection validation and uploads
(function(){
  function validateSelectionForModel(files, models, currentModel, visionEnabled) {
    const imageFiles = files.filter(f => (f.type || "").startsWith("image/"));
    const pdfFiles = files.filter(f => f.name.toLowerCase().endsWith('.pdf'));
    const nonPdfNonTextNonImage = files.filter(f => {
      const type = f.type || "";
      const name = f.name.toLowerCase();
      return !type.startsWith("text/") && !type.startsWith("image/") && !name.endsWith('.pdf');
    });
    
    if (imageFiles.length > 1) {
      alert("Only 1 image is allowed per message.");
      return { ok: false, reason: 'too-many-images' };
    }
    
    // PDFs are always allowed - they get indexed for RAG
    // Check if vision is enabled for images
    const isVision = visionEnabled;
    if (!isVision && imageFiles.length > 0) {
      alert("Vision is disabled. Remove images to proceed, or enable Force Vision.");
      return { ok: false, reason: 'images-on-text-only' };
    }
    
    // Block unsupported file types
    if (nonPdfNonTextNonImage.length > 0) {
      alert("Unsupported file type. Only text, images, and PDFs are allowed.");
      return { ok: false, reason: 'unsupported-type' };
    }
    
    return { ok: true };
  }

  async function uploadSelectedFiles(fileInput) {
    const fileList = document.getElementById("fileList");
    const files = Array.from(fileInput.files || []);
    if (files.length === 0) return [];
    const form = new FormData();
    for (const f of files) form.append("files", f);
    const res = await fetch("/upload", { method: "POST", body: form });
    if (!res.ok) throw new Error("Upload failed");
    const data = await res.json();
    for (const item of data.files || []) {
      if (!item.text && item.mime_type && item.mime_type.startsWith("text/")) {
        try {
          const t = await fetch(item.url).then(r => r.text());
          item.text = t.slice(0, 20000);
        } catch (_) { }
      }
      try { item.url = new URL(item.url, window.location.origin).toString(); } catch(_) {}
    }
    fileInput.value = "";
    fileList.textContent = "";
    return data.files || [];
  }

  window.Attach = {
    validateSelectionForModel,
    uploadSelectedFiles,
  };
})();
