// Chunked uploader for the RunPod proxy (Cloudflare caps a single request body near 100 MB).
// Splits the file into server-defined chunks, uploads 3 in parallel with retries, and resumes
// an interrupted upload of the same file (matched on name+size+lastModified via localStorage).
(function () {
  const PARALLEL = 3;
  const RETRIES = 4;
  const ROOT = "";

  function $(sel, root) { return (root || document).querySelector(sel); }

  function fmtMB(b) { return (b / 1048576).toFixed(1) + " MB"; }

  async function jsonFetch(url, opts) {
    const r = await fetch(url, Object.assign({ credentials: "same-origin" }, opts || {}));
    if (!r.ok) {
      let msg = r.status + " " + r.statusText;
      try { const j = await r.json(); if (j.detail) msg = j.detail; } catch (e) { /* ignore */ }
      throw new Error(msg);
    }
    return r.json();
  }

  async function putChunk(uploadId, index, blob) {
    let lastErr;
    for (let attempt = 0; attempt < RETRIES; attempt++) {
      try {
        return await jsonFetch(`${ROOT}/api/upload/chunk/${uploadId}/${index}`, {
          method: "PUT", body: blob, headers: { "Content-Type": "application/octet-stream" },
        });
      } catch (e) {
        lastErr = e;
        await new Promise(res => setTimeout(res, 1000 * Math.pow(2, attempt)));
      }
    }
    throw lastErr;
  }

  function storageKey(file) { return `enhance-upload:${file.name}:${file.size}:${file.lastModified}`; }

  async function startSession(file) {
    const key = storageKey(file);
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(key) || "null"); } catch (e) { /* ignore */ }
    if (saved && saved.upload_id) {
      try {
        const st = await jsonFetch(`${ROOT}/api/upload/status/${saved.upload_id}`);
        return { upload_id: saved.upload_id, chunk_size: st.chunk_size, total_chunks: st.total_chunks,
                 received: new Set(st.received), resumed: st.received.length > 0 };
      } catch (e) { /* stale session, fall through */ }
    }
    const init = await jsonFetch(`${ROOT}/api/upload/init`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file.name, size: file.size }),
    });
    try { localStorage.setItem(key, JSON.stringify({ upload_id: init.upload_id })); } catch (e) { /* ignore */ }
    return Object.assign(init, { received: new Set(), resumed: false });
  }

  async function uploadFile(file, ui) {
    ui.status.textContent = "Starting…";
    ui.bar.style.width = "0%";
    const s = await startSession(file);
    const pending = [];
    for (let i = 0; i < s.total_chunks; i++) if (!s.received.has(i)) pending.push(i);
    let doneChunks = s.total_chunks - pending.length;
    if (s.resumed) ui.status.textContent = `Resuming: ${doneChunks}/${s.total_chunks} chunks already uploaded`;
    const t0 = Date.now();
    let cursor = 0;
    async function worker() {
      while (cursor < pending.length) {
        const idx = pending[cursor++];
        const blob = file.slice(idx * s.chunk_size, Math.min(file.size, (idx + 1) * s.chunk_size));
        await putChunk(s.upload_id, idx, blob);
        doneChunks++;
        const sent = Math.min(file.size, doneChunks * s.chunk_size);
        const pct = Math.round(100 * doneChunks / s.total_chunks);
        const rate = sent / Math.max(0.001, (Date.now() - t0) / 1000);
        ui.bar.style.width = pct + "%";
        ui.status.textContent = `Uploading ${fmtMB(sent)} / ${fmtMB(file.size)} (${pct}%) – ${fmtMB(rate)}/s`;
      }
    }
    await Promise.all(Array.from({ length: Math.min(PARALLEL, pending.length || 1) }, worker));
    ui.status.textContent = "Assembling on server…";
    const res = await jsonFetch(`${ROOT}/api/upload/complete/${s.upload_id}`, { method: "POST" });
    try { localStorage.removeItem(storageKey(file)); } catch (e) { /* ignore */ }
    ui.bar.style.width = "100%";
    ui.status.textContent = `Done: ${res.name} (${fmtMB(file.size)})`;
    notifyGradio(res.path);
  }

  function notifyGradio(path) {
    const box = $("#uploaded-path textarea") || $("#uploaded-path input");
    if (!box) return;
    const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(box), "value").set;
    setter.call(box, path);
    box.dispatchEvent(new Event("input", { bubbles: true }));
    box.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function mount(root) {
    if (root.dataset.mounted) return;
    root.dataset.mounted = "1";
    root.innerHTML = `
      <div class="cu-wrap">
        <label class="cu-label">Large file upload (any size, chunked):
          <input type="file" class="cu-input" accept="video/*,.mp4,.mov,.3gp,.avi,.mkv,.m4v,.mts,.wmv,.webm">
        </label>
        <div class="cu-track"><div class="cu-bar"></div></div>
        <div class="cu-status">Pick a video to upload. Interrupted uploads resume when you re-select the same file.</div>
      </div>`;
    const ui = { bar: $(".cu-bar", root), status: $(".cu-status", root) };
    const input = $(".cu-input", root);
    input.addEventListener("change", async () => {
      const file = input.files && input.files[0];
      if (!file) return;
      input.disabled = true;
      try {
        await uploadFile(file, ui);
      } catch (e) {
        ui.status.textContent = "Upload failed: " + (e && e.message ? e.message : e) +
          " – re-select the same file to resume.";
      } finally {
        input.disabled = false;
        input.value = "";
      }
    });
  }

  function scan() {
    const root = document.getElementById("chunk-uploader");
    if (root) mount(root);
  }
  const obs = new MutationObserver(scan);
  obs.observe(document.documentElement, { childList: true, subtree: true });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", scan); else scan();
})();
