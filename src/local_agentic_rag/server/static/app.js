/* PDF Chat — vanilla JS single-page app. No frameworks, no CDNs: works offline and in Colab.
 *
 * Security rule for this file: text from documents, the model or the server is ONLY
 * ever inserted with textContent / createTextNode (via h()). innerHTML is never used.
 *
 * All URLs are relative ("api/...") so the app also works behind a path prefix
 * (Jupyter/Colab proxies).
 */
(() => {
  'use strict';

  // ================================================================ constants
  const LS_SELECTED = 'pdfchat.selected.v1';
  const LS_CHAT = 'pdfchat.chat.v1';
  const LS_ANSWER_LANG = 'pdfchat.answerLang.v1';
  const MAX_SAVED_MESSAGES = 60;
  const HISTORY_TURNS = 6;
  const MAX_PARALLEL_UPLOADS = 1; // the server allows one in-flight upload per address
  // Browsers allow ~6 HTTP/1.1 connections per origin; long-lived EventSources use them
  // up, so only the most important documents get a live stream and the rest are polled.
  const MAX_EVENT_STREAMS = 2;
  const ZOOM_LEVELS = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3];
  const DEFAULT_ZOOM_INDEX = 2;
  const PAGE_PT_WIDTH = 612; // US Letter / ~A4 width in PDF points, to pick a render zoom
  const MB = 1024 * 1024;

  const config = { maxUploadMb: 2048, maxQuestionChars: 2000, askTimeoutS: 300 };

  // ================================================================ tiny DOM helpers
  const $ = (id) => document.getElementById(id);

  /** h('div', {className, text, attrs, dataset, on}, ...children) — text only, never HTML. */
  function h(tag, props, ...children) {
    const el = document.createElement(tag);
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (v === undefined || v === null || v === false) continue;
        if (k === 'className') el.className = v;
        else if (k === 'text') el.textContent = v;
        else if (k === 'dataset') Object.assign(el.dataset, v);
        else if (k === 'on') for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
        else if (k === 'attrs') for (const [a, av] of Object.entries(v)) { if (av !== undefined && av !== null && av !== false) el.setAttribute(a, av === true ? '' : String(av)); }
        else el[k] = v;
      }
    }
    append(el, children);
    return el;
  }
  function append(el, children) {
    for (const c of children) {
      if (c === null || c === undefined || c === false) continue;
      if (Array.isArray(c)) append(el, c);
      else el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
  }

  const ICONS = {
    file: ['M7 3h7l5 5v11a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z', 'M14 3v5h5'],
    trash: ['M4 7h16', 'M10 11v6M14 11v6', 'M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12', 'M9 7V4h6v3'],
    retry: ['M4 12a8 8 0 0 1 14-5.3L20 9', 'M20 4v5h-5', 'M20 12a8 8 0 0 1-14 5.3L4 15', 'M4 20v-5h5'],
    copy: ['M9 9h10v10H9z', 'M5 15V5h10'],
    check: ['M5 12.5l4.5 4.5L19 7.5'],
    alert: ['M12 9v4', 'M12 17h.01', 'M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z'],
    search: ['M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14z', 'M20 20l-4-4'],
    shield: ['M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6z', 'M9 12l2 2 4-4'],
    x: ['M6 6l12 12M18 6 6 18'],
    spark: ['M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6'],
  };
  function icon(name, cls = 'icon') {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('class', cls);
    svg.setAttribute('aria-hidden', 'true');
    for (const d of ICONS[name] || []) {
      const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      p.setAttribute('d', d);
      svg.append(p);
    }
    return svg;
  }

  // ================================================================ formatting
  const nf = new Intl.NumberFormat();
  const fmtInt = (n) => nf.format(Math.round(n || 0));
  function fmtBytes(b) {
    if (!b) return '0 B';
    const u = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    while (b >= 1024 && i < u.length - 1) { b /= 1024; i += 1; }
    return `${b >= 100 || i === 0 ? Math.round(b) : b.toFixed(1)} ${u[i]}`;
  }
  function fmtDuration(s) {
    if (s === null || s === undefined || !isFinite(s)) return '';
    s = Math.max(0, Math.round(s));
    if (s < 10) return 'a few seconds';
    if (s < 60) return `${s} s`;
    const m = Math.round(s / 60);
    if (m < 60) return `${m} min`;
    return `${Math.floor(m / 60)} h ${m % 60} min`;
  }
  const fmtSecs = (s) => (s < 10 ? `${(+s).toFixed(1)} s` : `${Math.round(s)} s`);
  const plural = (n, one, many) => `${fmtInt(n)} ${n === 1 ? one : (many || one + 's')}`;
  const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36);

  // ================================================================ storage (best effort)
  const store = {
    get(key, fallback) {
      try { const raw = window.localStorage.getItem(key); return raw ? JSON.parse(raw) : fallback; } catch { return fallback; }
    },
    set(key, value) {
      try { window.localStorage.setItem(key, JSON.stringify(value)); return true; } catch { return false; }
    },
  };

  // ================================================================ API
  class ApiError extends Error {
    constructor(status, message, code) { super(message); this.status = status; this.code = code; }
  }
  async function api(path, { method = 'GET', body, signal } = {}) {
    const opts = { method, headers: { Accept: 'application/json' }, signal };
    if (body !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
    let res;
    try { res = await fetch(path, opts); } catch (e) {
      if (e && e.name === 'AbortError') throw e;
      throw new ApiError(0, 'Cannot reach the server. Check that it is still running.', 'network');
    }
    if (res.status === 204) return null;
    let data = null;
    try { data = await res.json(); } catch { /* not JSON */ }
    if (!res.ok) {
      const detail = data && typeof data.detail === 'string' ? data.detail : `Request failed (HTTP ${res.status}).`;
      throw new ApiError(res.status, detail);
    }
    return data;
  }
  const docUrl = (id, suffix = '') => `api/documents/${encodeURIComponent(id)}${suffix}`;

  // ---------------------------------------------------------------- access token (optional)
  // When the server runs with --token, /api answers 401 until the browser holds the session
  // cookie (set by POST api/session, HttpOnly, so this script never sees it). EventSource and
  // <img> cannot send an Authorization header, which is why a cookie is used. Nothing is
  // stored in the page: no localStorage, no sessionStorage.
  const rawFetch = window.fetch.bind(window);
  let loginPending = null;
  function requestLogin() {
    if (loginPending) return loginPending;
    loginPending = new Promise((resolve) => {
      const input = h('input', { id: 'auth-token', type: 'password', autocomplete: 'current-password', required: true, attrs: { 'aria-label': 'Access token' } });
      const error = h('p', { className: 'auth-error', attrs: { role: 'alert' }, hidden: true });
      const button = h('button', { type: 'submit', className: 'btn btn-primary', text: 'Unlock' });
      const form = h('form', { id: 'auth-form', className: 'auth-box' },
        h('h2', { className: 'section-title', text: 'Access token required' }),
        h('p', { className: 'muted', text: 'This server is protected. Enter its access token to continue.' }),
        input, error, button);
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        button.disabled = true;
        error.hidden = true;
        try {
          const res = await rawFetch('api/session', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token: input.value }),
          });
          if (res.ok) { overlay.remove(); loginPending = null; resolve(); return; }
          error.textContent = res.status === 401 ? 'That token is not correct.' : `Sign-in failed (HTTP ${res.status}).`;
        } catch { error.textContent = 'Cannot reach the server.'; }
        error.hidden = false;
        button.disabled = false;
        input.select();
      });
      const overlay = h('div', { className: 'auth-overlay', attrs: { role: 'dialog', 'aria-modal': 'true' } }, form);
      document.body.append(overlay);
      input.focus();
    });
    return loginPending;
  }
  // Every API call goes through here: on a 401, ask for the token once, then retry once.
  window.fetch = async (input, init) => {
    const res = await rawFetch(input, init);
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    if (res.status !== 401 || !/^(?:\.\/)?api\//.test(url) || /^(?:\.\/)?api\/session/.test(url)) return res;
    await requestLogin();
    return rawFetch(input, init);
  };

  /** Parse a text/event-stream body from fetch() into {event, data} objects. */
  async function* readSSE(body) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const raw = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          let event = 'message';
          const data = [];
          for (const line of raw.split('\n')) {
            const l = line.replace(/\r$/, '');
            if (!l || l.startsWith(':')) continue;
            const colon = l.indexOf(':');
            const field = colon < 0 ? l : l.slice(0, colon);
            const val = colon < 0 ? '' : l.slice(colon + 1).replace(/^ /, '');
            if (field === 'event') event = val;
            else if (field === 'data') data.push(val);
          }
          if (!data.length) continue;
          let parsed;
          try { parsed = JSON.parse(data.join('\n')); } catch { parsed = data.join('\n'); }
          yield { event, data: parsed };
        }
      }
    } finally {
      try { reader.releaseLock(); } catch { /* already released */ }
    }
  }

  // ================================================================ state
  const state = {
    docs: new Map(),            // doc_id -> DocRecord
    uploads: new Map(),         // local id -> upload job
    selected: new Set(store.get(LS_SELECTED, [])),
    messages: loadMessages(),
    pending: null,              // { messageId, controller, reason }
    confirmDelete: null,        // doc_id with the inline confirm open
    deleting: new Set(),
    streams: new Map(),         // doc_id -> { es, errors }
    health: null,
    serverReachable: true,
    loadedDocs: false,
    viewer: { docId: null, page: 1, chunkId: null, citePage: null, messageId: null, marker: null, zoomIndex: DEFAULT_ZOOM_INDEX },
  };

  function loadMessages() {
    const msgs = store.get(LS_CHAT, []);
    if (!Array.isArray(msgs)) return [];
    for (const m of msgs) {
      if (m && m.role === 'assistant' && m.status === 'pending') {
        m.status = 'error';
        m.error = { code: 'interrupted', detail: 'The page was reloaded before this answer arrived.' };
      }
    }
    return msgs.filter((m) => m && (m.role === 'user' || m.role === 'assistant'));
  }
  function saveMessages() {
    let msgs = state.messages.slice(-MAX_SAVED_MESSAGES);
    if (store.set(LS_CHAT, msgs)) return;
    // Quota exceeded: drop traces, then halve until it fits.
    msgs = msgs.map((m) => (m.answer ? { ...m, answer: { ...m.answer, trace: [] } } : m));
    while (msgs.length && !store.set(LS_CHAT, msgs)) msgs = msgs.slice(Math.ceil(msgs.length / 2));
  }
  const saveSelected = () => store.set(LS_SELECTED, [...state.selected]);

  // "Answer in" menu: automatic (null) or one language for every answer, remembered here.
  function answerLanguage() {
    const v = $('answer-lang') ? $('answer-lang').value : 'auto';
    return v && v !== 'auto' ? v : null;
  }
  function initAnswerLanguage() {
    const sel = $('answer-lang');
    if (!sel) return;
    const saved = store.get(LS_ANSWER_LANG, 'auto');
    if ([...sel.options].some((o) => o.value === saved)) sel.value = saved;
    sel.addEventListener('change', () => store.set(LS_ANSWER_LANG, sel.value));
  }

  const readyDocs = () => [...state.docs.values()].filter((d) => d.status === 'ready');
  /** Ready, or still processing but with an indexed prefix that can already answer (partial answer). */
  const isAskable = (d) => !!d && (d.status === 'ready' || (d.status === 'processing' && d.askable === true));
  const askableDocs = () => [...state.docs.values()].filter(isAskable);
  const sortedDocs = () => [...state.docs.values()].sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  /** Ids the next question is about, or null = every ready document (processing ones only when ticked). */
  function scopeIds() {
    const ids = [...state.selected].filter((id) => isAskable(state.docs.get(id)));
    return ids.length ? ids : null;
  }

  // ================================================================ toasts
  function toast(message, kind = 'info', ms = 4500) {
    const t = h('div', { className: `toast${kind === 'error' ? ' is-error' : ''}`, text: message });
    $('toasts').append(t);
    setTimeout(() => t.remove(), ms);
  }

  // ================================================================ library rendering
  const docNodes = new Map(); // key -> {li, ...refs}

  function renderLibrary() {
    const list = $('doc-list');
    const items = [];
    for (const u of state.uploads.values()) items.push(uploadItem(u));
    for (const d of sortedDocs()) items.push(docItem(d));
    const keep = new Set(items);
    for (const [key, n] of docNodes) if (!keep.has(n.li)) { n.li.remove(); docNodes.delete(key); }
    items.forEach((li, i) => { if (list.children[i] !== li) list.insertBefore(li, list.children[i] || null); });

    const empty = state.loadedDocs && items.length === 0;
    $('empty-library').hidden = !empty;
    const n = state.docs.size;
    const badge = $('tab-doc-count');
    badge.hidden = n === 0;
    badge.textContent = String(n);
    const sel = $('select-toggle');
    sel.hidden = state.selected.size === 0;
    sel.textContent = 'Clear selection';
    renderScope();
  }

  function docItem(doc) {
    const key = `d:${doc.doc_id}`;
    let n = docNodes.get(key);
    if (!n) {
      n = {};
      n.input = h('input', { type: 'checkbox', on: { change: () => toggleSelect(doc.doc_id, n.input.checked) } });
      n.title = h('div', { className: 'doc-title' });
      n.pill = h('span', { className: 'pill' });
      n.meta = h('div', { className: 'doc-meta' });
      n.bar = h('div', { className: 'progress-bar' });
      n.progress = h('div', { className: 'progress', attrs: { role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100 } }, n.bar);
      n.stage = h('div', { className: 'doc-stage' });
      n.error = h('div', { className: 'doc-error', attrs: { role: 'alert' } });
      n.warnings = h('details', { className: 'doc-warnings' });
      n.actions = h('div', { className: 'doc-actions' });
      n.body = h('div', { className: 'doc-body' }, n.progress, n.stage, n.error, n.warnings, n.actions);
      n.li = h('li', { className: 'doc' },
        h('label', { className: 'doc-check' }, n.input),
        h('div', null, n.title, n.meta),
        n.body);
      n.actionsKey = '';
      docNodes.set(key, n);
    }
    const ready = doc.status === 'ready';
    const canAsk = isAskable(doc);
    const selected = state.selected.has(doc.doc_id);
    n.li.classList.toggle('is-selected', selected && canAsk);
    n.input.checked = selected;
    n.input.disabled = !canAsk;
    n.input.setAttribute('aria-label', ready ? `Ask about ${doc.title}` : canAsk ? `Ask about the first pages of ${doc.title}` : `${doc.title} (not ready yet)`);
    n.input.title = ready ? 'Limit questions to the ticked documents'
      : canAsk ? `Ask about the first ${fmtInt(doc.pages_indexed)} pages now; the rest is still being processed` : 'Available once the first pages are indexed';
    n.title.textContent = doc.title || doc.filename;
    n.title.title = doc.filename;

    // status pill + meta line
    const pct = doc.num_pages ? Math.min(100, Math.floor((100 * doc.pages_done) / doc.num_pages)) : null;
    const pillText = { queued: 'Queued', processing: pct === null ? 'Processing' : `Processing ${pct}%`, ready: 'Ready', failed: 'Failed' }[doc.status] || doc.status;
    n.pill.className = `pill pill-${doc.status}`;
    n.pill.textContent = pillText;
    const meta = [n.pill];
    if (doc.num_pages) meta.push(h('span', { text: plural(doc.num_pages, 'page') }));
    if (doc.size_bytes) meta.push(h('span', { text: fmtBytes(doc.size_bytes) }));
    if (ready && doc.ocr_pages) meta.push(h('span', { text: `${fmtInt(doc.ocr_pages)} OCR`, title: `${plural(doc.ocr_pages, 'page')} read with OCR (scanned)` }));
    if (ready && doc.ingest_seconds) meta.push(h('span', { text: `in ${fmtDuration(doc.ingest_seconds)}`, title: 'Processing time' }));
    n.meta.replaceChildren(...meta);

    // progress
    const active = doc.status === 'processing';
    n.progress.hidden = !active;
    if (active) {
      n.progress.classList.toggle('is-indeterminate', pct === null);
      n.bar.style.width = pct === null ? '' : `${Math.max(2, pct)}%`;
      if (pct === null) n.progress.removeAttribute('aria-valuenow'); else n.progress.setAttribute('aria-valuenow', pct);
      n.progress.setAttribute('aria-label', `Processing ${doc.title}`);
    }
    const stageBits = [];
    if (doc.status === 'queued') stageBits.push(doc.stage || 'Waiting for the documents ahead of it');
    if (active) {
      stageBits.push(doc.stage || 'Processing…');
      if (doc.eta_seconds) stageBits.push(`about ${fmtDuration(doc.eta_seconds)} left`);
      if (doc.askable && doc.pages_indexed) stageBits.push(`first ${fmtInt(doc.pages_indexed)} pages can be asked about`);
    }
    n.stage.hidden = stageBits.length === 0;
    n.stage.textContent = stageBits.join(' · ');
    n.error.hidden = !(doc.status === 'failed' && doc.error);
    n.error.textContent = doc.error || '';

    const warns = Array.isArray(doc.warnings) ? doc.warnings : [];
    n.warnings.hidden = warns.length === 0;
    if (warns.length) {
      const wkey = JSON.stringify(warns);
      if (n.warnings.dataset.key !== wkey) {
        n.warnings.dataset.key = wkey;
        n.warnings.replaceChildren(
          h('summary', { text: warns.length === 1 ? '1 note' : `${warns.length} notes` }),
          h('ul', null, warns.map((w) => h('li', { text: w }))));
      }
    }

    // actions (rebuilt only when their state changes, so focus is not lost on progress ticks)
    const confirming = state.confirmDelete === doc.doc_id;
    const deleting = state.deleting.has(doc.doc_id);
    const akey = `${doc.status}|${confirming}|${deleting}`;
    if (akey !== n.actionsKey) {
      const hadFocus = n.actions.contains(document.activeElement);
      n.actionsKey = akey;
      if (confirming) {
        const cancel = h('button', { type: 'button', className: 'btn btn-sm', text: 'Cancel', on: { click: () => setConfirm(null, doc.doc_id) } });
        const del = h('button', {
          type: 'button', className: 'btn btn-danger btn-sm', disabled: deleting,
          text: deleting ? 'Deleting…' : 'Delete', on: { click: () => deleteDoc(doc.doc_id) },
        });
        n.actions.replaceChildren(h('div', { className: 'doc-confirm', attrs: { role: 'group', 'aria-label': 'Confirm delete' } },
          h('span', { text: doc.status === 'processing' ? 'Stop processing and delete this document?' : 'Delete this document and its index?' }),
          del, cancel));
        if (!deleting) cancel.focus();
      } else {
        const btns = [];
        if (doc.status === 'failed') {
          btns.push(h('button', { type: 'button', className: 'btn btn-sm', on: { click: () => retryDoc(doc.doc_id) } }, icon('retry', 'icon icon-sm'), 'Retry'));
        }
        btns.push(h('button', {
          type: 'button', className: 'btn btn-ghost btn-sm', dataset: { role: 'delete' },
          attrs: { 'aria-label': `Delete ${doc.title}` }, on: { click: () => setConfirm(doc.doc_id) },
        }, icon('trash', 'icon icon-sm'), 'Delete'));
        n.actions.replaceChildren(...btns);
        if (hadFocus) n.actions.querySelector('[data-role="delete"]')?.focus();
      }
    }
    return n.li;
  }

  function uploadItem(u) {
    const key = `u:${u.id}`;
    let n = docNodes.get(key);
    if (!n) {
      n = {};
      n.title = h('div', { className: 'doc-title', text: u.file.name });
      n.pill = h('span', { className: 'pill' });
      n.meta = h('div', { className: 'doc-meta' });
      n.bar = h('div', { className: 'progress-bar' });
      n.progress = h('div', { className: 'progress', attrs: { role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-label': `Uploading ${u.file.name}` } }, n.bar);
      n.error = h('div', { className: 'doc-error', attrs: { role: 'alert' } });
      n.actions = h('div', { className: 'doc-actions' });
      n.li = h('li', { className: 'doc is-upload' },
        h('div', { className: 'doc-check' }, icon('file', 'icon doc-icon')),
        h('div', null, n.title, n.meta),
        h('div', { className: 'doc-body' }, n.progress, n.error, n.actions));
      n.actionsKey = '';
      docNodes.set(key, n);
    }
    const pct = u.total ? Math.floor((100 * u.loaded) / u.total) : 0;
    const label = { waiting: 'Waiting', uploading: `Uploading ${pct}%`, finishing: 'Finishing…', error: 'Upload failed' }[u.status];
    n.pill.className = `pill ${u.status === 'error' ? 'pill-failed' : 'pill-uploading'}`;
    n.pill.textContent = label;
    const metaText = u.status === 'uploading' ? `${fmtBytes(u.loaded)} of ${fmtBytes(u.total)}` : fmtBytes(u.total);
    n.meta.replaceChildren(n.pill, h('span', { text: metaText }));
    n.progress.hidden = u.status === 'error';
    n.progress.classList.toggle('is-indeterminate', u.status === 'finishing');
    n.bar.style.width = u.status === 'finishing' ? '' : `${Math.max(2, pct)}%`;
    n.progress.setAttribute('aria-valuenow', pct);
    n.error.hidden = u.status !== 'error';
    n.error.textContent = u.error || '';
    const akey = u.status;
    if (akey !== n.actionsKey) {
      n.actionsKey = akey;
      if (u.status === 'error') {
        n.actions.replaceChildren(
          h('button', { type: 'button', className: 'btn btn-sm', on: { click: () => { u.status = 'waiting'; u.error = null; u.loaded = 0; pumpUploads(); renderLibrary(); } } }, icon('retry', 'icon icon-sm'), 'Try again'),
          h('button', { type: 'button', className: 'btn btn-ghost btn-sm', text: 'Dismiss', on: { click: () => { state.uploads.delete(u.id); renderLibrary(); } } }));
      } else {
        n.actions.replaceChildren(h('button', {
          type: 'button', className: 'btn btn-ghost btn-sm', text: 'Cancel',
          attrs: { 'aria-label': `Cancel upload of ${u.file.name}` },
          on: { click: () => { if (u.xhr) u.xhr.abort(); state.uploads.delete(u.id); renderLibrary(); pumpUploads(); } },
        }));
      }
    }
    return n.li;
  }

  function renderScope() {
    const ready = readyDocs();
    const ids = scopeIds();
    const chip = $('scope-chip');
    const hint = $('scope-hint');
    if (!ready.length && !ids) {
      const processing = [...state.docs.values()].some((d) => d.status === 'queued' || d.status === 'processing');
      chip.replaceChildren(icon('file', 'icon icon-sm'), h('span', { text: processing ? 'Your documents are being processed…' : 'No documents yet' }));
      hint.textContent = askableDocs().length ? 'Tick a document that is still processing to ask about its first pages.'
        : state.docs.size ? 'Documents can be asked about once they are ready.' : '';
    } else if (!ids) {
      chip.replaceChildren(icon('search', 'icon icon-sm'), 'Searching ', h('strong', { text: ready.length === 1 ? ready[0].title : `all ${ready.length} documents` }));
      hint.textContent = ready.length > 1 ? 'Tick documents to limit questions to them.' : '';
    } else {
      const names = ids.map((id) => state.docs.get(id).title);
      chip.replaceChildren(icon('search', 'icon icon-sm'), 'Searching ', h('strong', { text: ids.length === 1 ? names[0] : `${ids.length} of ${askableDocs().length} documents`, title: names.join(', ') }));
      hint.textContent = `${plural(ids.length, 'document')} selected for questions.`;
    }
    renderWelcome();
  }

  function toggleSelect(id, on) {
    if (on) state.selected.add(id); else state.selected.delete(id);
    saveSelected();
    renderLibrary();
  }

  function setConfirm(id, restoreFocusFor) {
    state.confirmDelete = id;
    renderLibrary();
    if (!id && restoreFocusFor) docNodes.get(`d:${restoreFocusFor}`)?.actions.querySelector('[data-role="delete"]')?.focus();
  }

  // ================================================================ library actions
  function upsertDoc(rec) {
    if (!rec || !rec.doc_id) return;
    state.docs.set(rec.doc_id, rec);
    renderLibrary();
    ensureWatchers();
    if (state.viewer.docId === rec.doc_id) updateViewerChrome();
  }
  function removeDoc(id) {
    state.docs.delete(id);
    stopWatch(id);
    if (state.selected.delete(id)) saveSelected();
    if (state.confirmDelete === id) state.confirmDelete = null;
    if (state.viewer.docId === id) showViewerError('This document is no longer in the library.');
    renderLibrary();
  }

  async function loadDocs() {
    try {
      const data = await api('api/documents');
      const seen = new Set();
      for (const rec of data.documents || []) { seen.add(rec.doc_id); state.docs.set(rec.doc_id, rec); }
      for (const id of [...state.docs.keys()]) if (!seen.has(id) && !state.deleting.has(id)) removeDoc(id);
      if (!state.loadedDocs) {
        state.loadedDocs = true;
        let pruned = false;
        for (const id of [...state.selected]) if (!state.docs.has(id)) { state.selected.delete(id); pruned = true; }
        if (pruned) saveSelected();
      }
      setReachable(true);
      renderLibrary();
      ensureWatchers();
    } catch (e) {
      if (e.code === 'network') setReachable(false);
    }
  }

  // Polls the whole list: picks up documents that have no live stream (queued ones, or when
  // EventSource is unavailable) and changes made from other tabs or by other users.
  let refreshTimer = null;
  let refreshDue = Infinity;
  function scheduleRefresh() {
    const busy = [...state.docs.values()].some((d) => (d.status === 'queued' || d.status === 'processing') && !state.streams.has(d.doc_id));
    const due = Date.now() + (busy ? 3000 : 20000);
    if (refreshTimer && refreshDue <= due) return; // an earlier refresh is already scheduled
    clearTimeout(refreshTimer);
    refreshDue = due;
    refreshTimer = setTimeout(async () => {
      refreshTimer = null;
      refreshDue = Infinity;
      await loadDocs();
      scheduleRefresh();
    }, due - Date.now());
  }

  function ensureWatchers() {
    const active = [...state.docs.values()].filter((d) => d.status === 'queued' || d.status === 'processing');
    for (const [id] of state.streams) if (!active.some((d) => d.doc_id === id)) stopWatch(id);
    // processing first: those change every second; queued ones are covered by polling
    active.sort((a, b) => (a.status === b.status ? 0 : a.status === 'processing' ? -1 : 1));
    for (const d of active) {
      if (state.streams.size >= MAX_EVENT_STREAMS) break;
      if (!state.streams.has(d.doc_id) && d.status === 'processing') watchDoc(d.doc_id);
    }
    scheduleRefresh();
  }

  function watchDoc(id) {
    if (typeof EventSource === 'undefined') return;
    const es = new EventSource(docUrl(id, '/events'));
    const entry = { es, errors: 0 };
    state.streams.set(id, entry);
    es.addEventListener('document', (e) => {
      entry.errors = 0;
      try { upsertDoc(JSON.parse(e.data)); } catch { /* ignore malformed */ }
    });
    es.addEventListener('done', () => { stopWatch(id); ensureWatchers(); });
    es.addEventListener('deleted', () => { stopWatch(id); removeDoc(id); });
    es.onerror = () => {
      entry.errors += 1;
      // EventSource reconnects by itself; give up after repeated failures and fall back to polling.
      if (es.readyState === EventSource.CLOSED || entry.errors > 4) { stopWatch(id); scheduleRefresh(); }
    };
  }
  function stopWatch(id) {
    const entry = state.streams.get(id);
    if (entry) { entry.es.close(); state.streams.delete(id); }
  }

  async function deleteDoc(id) {
    state.deleting.add(id);
    renderLibrary();
    try {
      await api(docUrl(id), { method: 'DELETE' });
      const title = state.docs.get(id)?.title || 'Document';
      state.deleting.delete(id);
      removeDoc(id);
      toast(`Deleted “${title}”.`);
    } catch (e) {
      state.deleting.delete(id);
      if (e.status === 404) { removeDoc(id); return; }
      state.confirmDelete = null;
      renderLibrary();
      toast(e.message, 'error');
    }
  }

  async function retryDoc(id) {
    try { upsertDoc(await api(docUrl(id, '/retry'), { method: 'POST' })); } catch (e) { toast(e.message, 'error'); }
  }

  // ================================================================ uploads
  function addFiles(fileList) {
    const files = [...(fileList || [])];
    for (const file of files) {
      const isPdf = /\.pdf$/i.test(file.name) || file.type === 'application/pdf';
      if (!isPdf) { toast(`“${file.name}” is not a PDF.`, 'error'); continue; }
      if (file.size === 0) { toast(`“${file.name}” is empty.`, 'error'); continue; }
      if (file.size > config.maxUploadMb * MB) {
        toast(`“${file.name}” is ${fmtBytes(file.size)} — the limit is ${fmtBytes(config.maxUploadMb * MB)}.`, 'error', 7000);
        continue;
      }
      const u = { id: uid(), file, loaded: 0, total: file.size, status: 'waiting', error: null, xhr: null };
      state.uploads.set(u.id, u);
    }
    renderLibrary();
    pumpUploads();
    if (files.length && isNarrow()) setView('library');
  }

  function pumpUploads() {
    let running = [...state.uploads.values()].filter((u) => u.status === 'uploading' || u.status === 'finishing').length;
    for (const u of state.uploads.values()) {
      if (running >= MAX_PARALLEL_UPLOADS) break;
      if (u.status === 'waiting') { startUpload(u); running += 1; }
    }
  }

  // Ask the server whether this file would be accepted BEFORE sending its body. A server that
  // rejects a large upload early (too big, over quota, another upload running) cannot get its
  // answer through reliably: browsers and proxies often report a dropped connection instead and
  // the reason is lost. Returns the reason text, or null when the upload may go ahead (also when
  // the check itself cannot be made: the upload will then report whatever goes wrong).
  async function uploadBlocker(u) {
    let res;
    try { res = await fetch(`api/upload-check?size=${u.file.size}`, { headers: { Accept: 'application/json' } }); } catch { return null; }
    if (res.ok) return null;
    let data = null;
    try { data = await res.json(); } catch { /* not JSON */ }
    return data && typeof data.detail === 'string' ? data.detail : null;
  }

  function startUpload(u) {
    u.status = 'uploading';
    u.loaded = 0;
    uploadBlocker(u).then((reason) => {
      if (state.uploads.get(u.id) !== u) return; // cancelled while checking
      if (reason) { u.status = 'error'; u.error = reason; u.xhr = null; renderLibrary(); pumpUploads(); return; }
      sendUpload(u);
    });
    renderLibrary();
  }

  function sendUpload(u) {
    const xhr = new XMLHttpRequest();
    u.xhr = xhr;
    xhr.open('POST', 'api/documents');
    xhr.setRequestHeader('Accept', 'application/json');
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) { u.loaded = e.loaded; u.total = e.total; }
      renderLibrary();
    };
    xhr.upload.onload = () => { u.status = 'finishing'; renderLibrary(); };
    const fail = (msg) => { u.status = 'error'; u.error = msg; u.xhr = null; renderLibrary(); pumpUploads(); };
    xhr.onload = () => {
      if (xhr.status === 401) { requestLogin().then(() => startUpload(u)); return; } // token needed: retry after sign-in
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300 && data && data.doc_id) {
        const existed = data.already_in_library === true || state.docs.has(data.doc_id);
        state.uploads.delete(u.id);
        upsertDoc(data);
        if (data.status === 'failed') toast(`“${data.title}” could not be processed${data.error ? `: ${data.error}` : '.'}`, 'error', 8000);
        else if (existed) toast(`“${data.title}” is already in your library${data.status === 'ready' ? '' : ' (still being processed)'}.`);
        else toast(`“${data.title}” uploaded — processing has started.`);
        pumpUploads();
      } else if (data && typeof data.detail === 'string') {
        fail(data.detail);
      } else if (xhr.status === 413) {
        fail('The server, or a proxy in front of it, refused the file as too large.');
      } else {
        fail(`Upload failed (HTTP ${xhr.status}).`);
      }
    };
    // No response at all. If the server rejected the upload early, the reason was lost with the
    // connection: ask again (it is cheap) and show the real reason when there is one.
    xhr.onerror = () => uploadBlocker(u).then((reason) => {
      if (state.uploads.get(u.id) !== u) return;
      fail(reason || (u.loaded >= u.total && u.total > 0
        ? 'The connection closed before the server answered, so it is unknown whether the file was accepted. Check the library, then try again.'
        : 'The connection dropped during the upload. Check your connection and try again.'));
    });
    xhr.onabort = () => { u.xhr = null; };
    const form = new FormData();
    form.append('file', u.file, u.file.name);
    xhr.send(form);
    renderLibrary();
  }

  // ================================================================ chat rendering
  const msgNodes = new Map();

  // The log is a polite live region. While an answer is still being written its card is
  // rebuilt on every token, so mark the region busy: screen readers then stay quiet and
  // announce the finished answer once, when busy is cleared.
  function syncLiveBusy() {
    $('messages').setAttribute('aria-busy', String(state.messages.some((x) => x.status === 'pending')));
  }

  function renderMessages() {
    const box = $('messages');
    for (const [id, node] of msgNodes) if (!state.messages.some((m) => m.id === id)) { node.remove(); msgNodes.delete(id); }
    for (const m of state.messages) {
      if (!msgNodes.has(m.id)) { const node = buildMessage(m); msgNodes.set(m.id, node); box.append(node); }
    }
    syncLiveBusy();
    renderWelcome();
  }

  function updateMessage(m) {
    if (!state.messages.includes(m)) return; // e.g. the chat was cleared while it was pending
    const old = msgNodes.get(m.id);
    const node = buildMessage(m);
    msgNodes.set(m.id, node);
    if (old) old.replaceWith(node); else $('messages').append(node);
    syncLiveBusy();
    renderWelcome();
  }

  function renderWelcome() {
    const welcome = $('welcome');
    welcome.hidden = state.messages.length > 0;
    if (welcome.hidden) return;
    const ready = readyDocs();
    const sub = $('welcome-sub');
    const sugg = $('suggestions');
    if (!state.docs.size) {
      sub.textContent = 'Upload a PDF on the left. Every answer cites the pages it comes from, and you can open each source to check it.';
      sugg.replaceChildren();
    } else if (!ready.length && !askableDocs().length) {
      sub.textContent = 'Your documents are being processed. You can ask questions as soon as one is ready.';
      sugg.replaceChildren();
    } else {
      sub.textContent = 'Every answer cites the pages it comes from — click a citation to see the highlighted passage.';
      const ideas = ['Summarise the main points', 'What are the key numbers and dates?', 'What conclusions does it reach?'];
      sugg.replaceChildren(...ideas.map((q) => h('button', { type: 'button', className: 'btn suggestion', text: q, on: { click: () => send(q) } })));
    }
  }

  function scrollToBottom(force) {
    const box = $('messages');
    const near = box.scrollHeight - box.scrollTop - box.clientHeight < 160;
    if (force || near) box.scrollTop = box.scrollHeight;
  }

  function buildMessage(m) {
    if (m.role === 'user') {
      return h('div', { className: 'msg msg-user' }, h('div', { className: 'bubble', text: m.text }));
    }
    const wrap = h('div', { className: 'msg msg-assistant', dataset: { id: m.id } });
    if (m.status === 'pending') wrap.append(buildPending(m));
    else if (m.status === 'error') wrap.append(buildError(m));
    else wrap.append(buildAnswer(m));
    return wrap;
  }

  // ---- live "thinking" indicator
  const STEP_DONE = {
    plan: 'Planned the search', retrieve: 'Searched', grade: 'Checked relevance', generate: 'Drafted answer',
    verify: 'Checked the answer', escalate: 'Searching again', refuse: 'No match', calc: 'Calculated',
    calc_appended: 'Added calculation', infer_citations: 'Matched citations', keep_first_answer: 'Kept first answer',
    table_reference_added: 'Named the table',
  };
  const STEP_NEXT = {
    plan: 'Searching your documents…', retrieve: 'Reading the passages and writing an answer…',
    grade: 'Writing the answer…', generate: 'Verifying the answer against the sources…',
    verify: 'Finishing up…', escalate: 'Searching again, more broadly…', calc: 'Writing the answer…',
    refuse: 'Finishing up…', infer_citations: 'Finishing up…', keep_first_answer: 'Finishing up…',
  };
  const stepLabel = (s) => STEP_DONE[s] || String(s).replace(/_/g, ' ');

  function thinkingText(m) {
    if (m.phase === 'queued') return m.queuePos > 1 ? `Waiting for a free slot — you are #${m.queuePos} in line…` : 'Waiting for a free slot — you are next…';
    if (m.phase === 'connecting') return 'Sending your question…';
    const last = m.steps && m.steps.length ? m.steps[m.steps.length - 1].step : null;
    return (last && STEP_NEXT[last]) || (last ? 'Working…' : 'Understanding your question…');
  }

  function buildPending(m) {
    if (m.partial && !isSentinelText(m.partial)) return buildStreaming(m);
    const steps = (m.steps || []).map((s) => {
      let text = stepLabel(s.step);
      const pages = s.detail && Array.isArray(s.detail.pages) ? s.detail.pages : null;
      if (s.step === 'retrieve' && pages && pages.length) text += ` · p. ${[...new Set(pages)].slice(0, 4).join(', ')}`;
      return h('li', null, icon('check', 'icon icon-sm'), text);
    });
    return h('div', { className: 'card thinking', attrs: { 'aria-busy': 'true' } },
      h('div', { className: 'thinking-now' }, h('span', { className: 'dots', attrs: { 'aria-hidden': 'true' } }, h('span'), h('span'), h('span')),
        h('span', { text: thinkingText(m) })),
      steps.length ? h('ol', { className: 'thinking-steps', attrs: { 'aria-label': 'Steps completed' } }, steps) : null,
      h('div', { className: 'thinking-elapsed', dataset: { since: m.startedAt } }, elapsedText(m.startedAt)));
  }
  // A refusal is generated as the bare marker ANSWER_NOT_FOUND; never flash that to the reader.
  const isSentinelText = (t) => { const x = t.trim(); return x.includes('ANSWER_NOT_FOUND') || 'ANSWER_NOT_FOUND'.startsWith(x); };
  // Live answer text while the model is still writing. It is shown as plain DOM text (never
  // raw HTML) with no citation chips: those need the whole text and come with the final answer.
  function buildStreaming(m) {
    const status = h('div', { className: 'thinking-now stream-check' },
      h('span', { className: 'dots', attrs: { 'aria-hidden': 'true' } }, h('span'), h('span'), h('span')),
      h('span', { text: m.streamDone ? 'Draft — checking it against the sources, it may change…' : 'Draft — still being written, not checked yet…' }));
    return h('div', { className: `card thinking streaming${m.streamDone ? ' is-checking' : ''}`, attrs: { 'aria-busy': 'true', 'aria-live': 'off' } },
      renderRichText(m.partial, () => null),
      status,
      h('div', { className: 'thinking-elapsed', dataset: { since: m.startedAt } }, elapsedText(m.startedAt)));
  }
  const elapsedText = (since) => `${Math.floor((Date.now() - since) / 1000)} s`;
  setInterval(() => {
    for (const el of document.querySelectorAll('.thinking-elapsed')) el.textContent = elapsedText(+el.dataset.since);
  }, 1000);

  // ---- answer text: paragraphs, lists, **bold**, `code`, and [n] citation chips
  function renderRichText(text, citeFor) {
    const root = h('div', { className: 'answer-text' });
    const blocks = String(text || '').replace(/\r\n?/g, '\n').split(/\n\s*\n/);
    for (const block of blocks) {
      const lines = block.split('\n').filter((l) => l.trim());
      if (!lines.length) continue;
      const bullet = /^\s*[-*•]\s+/;
      const numbered = /^\s*\d+[.)]\s+/;
      if (lines.every((l) => bullet.test(l)) || lines.every((l) => numbered.test(l))) {
        const ordered = numbered.test(lines[0]);
        root.append(h(ordered ? 'ol' : 'ul', null, lines.map((l) => h('li', null, inline(l.replace(ordered ? numbered : bullet, ''), citeFor)))));
      } else {
        const p = h('p');
        lines.forEach((l, i) => { if (i) p.append(h('br')); append(p, inline(l, citeFor)); });
        root.append(p);
      }
    }
    return root;
  }
  function inline(text, citeFor) {
    const out = [];
    const re = /(\*\*[^*\n]+\*\*|`[^`\n]+`|\[\d+(?:\s*[,;]\s*\d+)*\])/g;
    let last = 0;
    let m;
    while ((m = re.exec(text))) {
      if (m.index > last) out.push(text.slice(last, m.index));
      const tok = m[0];
      if (tok.startsWith('**')) out.push(h('strong', { text: tok.slice(2, -2) }));
      else if (tok.startsWith('`')) out.push(h('code', { text: tok.slice(1, -1) }));
      else {
        const nums = tok.slice(1, -1).split(/[,;]/).map((s) => parseInt(s, 10));
        const chips = nums.map((num) => citeFor(num));
        if (chips.every(Boolean)) out.push(...chips); else out.push(tok);
      }
      last = re.lastIndex;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  }

  function citationChip(m, c) {
    const doc = state.docs.get(c.doc_id);
    const active = state.viewer.messageId === m.id && state.viewer.marker === c.marker;
    return h('button', {
      type: 'button', className: `chip${active ? ' is-active' : ''}`,
      dataset: { msg: m.id, marker: c.marker },
      attrs: { 'aria-label': `Source ${c.marker}: ${doc ? doc.title + ', ' : ''}page ${c.page}`, title: c.snippet || '' },
      on: { click: () => openCitation(m.id, c.marker) },
    }, `[${c.marker}]`, ...(manyDocs(m) && doc ? [h('span', { className: 'chip-doc', text: doc.title })] : []),
    h('span', { className: 'chip-page', text: `p.${c.page}` }));
  }

  // Name the book on each citation when the question searched more than one.
  function manyDocs(m) {
    const ids = m.docIds && m.docIds.length ? m.docIds : askableDocs().map((d) => d.doc_id);
    return ids.length > 1;
  }

  function buildAnswer(m) {
    const a = m.answer || {};
    const cites = Array.isArray(a.citations) ? a.citations : [];
    const citeFor = (num) => { const c = cites.find((x) => x.marker === num); return c ? citationChip(m, c) : null; };
    const card = h('div', { className: `card${a.refused ? ' is-refused' : ''}` });
    if (a.partial && a.partial.pages_total) {
      // Answers about a document that is still being processed only saw its first pages.
      card.append(h('div', { className: 'partial-banner', attrs: { role: 'status' } }, icon('alert', 'icon icon-sm'),
        h('span', { text: `Answered from the first ${fmtInt(a.partial.pages_indexed)} of ${fmtInt(a.partial.pages_total)} pages; this document is still being processed.` })));
    }

    if (a.error) {
      card.classList.add('is-error');
      card.append(h('div', { className: 'error-head' }, icon('alert'), 'The answer could not be completed'),
        h('p', { text: a.error }));
      if (a.answer && !a.refused) card.append(renderRichText(a.answer, citeFor));
      card.append(h('div', { className: 'answer-foot' }, retryButton(m)));
      return card;
    }
    if (a.refused) {
      const sawAll = !(a.partial && a.partial.pages_total);
      card.append(h('div', { className: 'refusal-head' }, icon('search'), sawAll ? 'Not found in your documents' : `Not found in the first ${fmtInt(a.partial.pages_indexed)} pages`),
        renderRichText(a.answer || 'I could not find the answer to this question in the document.', citeFor),
        h('p', { className: 'refusal-tip', text: 'Try rephrasing the question, or select different documents on the left.' }));
    } else {
      card.append(renderRichText(a.answer, citeFor));
    }

    if (a.grounded === false && Array.isArray(a.unsupported_claims) && a.unsupported_claims.length) {
      card.append(h('details', { className: 'unsupported' },
        h('summary', { text: `${plural(a.unsupported_claims.length, 'statement')} could not be verified` }),
        h('ul', null, a.unsupported_claims.map((s) => h('li', { text: s })))));
    }

    const foot = h('div', { className: 'answer-foot' });
    if (!a.refused) {
      if (a.grounded === false) foot.append(h('span', { className: 'badge badge-warn', title: 'Some statements could not be checked against the document. Open the sources to verify.' }, icon('alert', 'icon icon-sm'), 'Unverified'));
      else if (a.grounded === true) foot.append(h('span', { className: 'badge badge-ok', title: 'A second pass checked the answer against the passages it was based on.' }, icon('shield', 'icon icon-sm'), 'Verified'));
      else foot.append(h('span', { className: 'badge', title: 'The automatic check did not run or gave no result. Open the sources to verify.' }, 'Not checked'));
    }
    const inferred = Array.isArray(a.trace) && a.trace.some((t) => t.step === 'infer_citations');
    if (cites.length) foot.append(h('span', { text: inferred ? 'Sources (matched by wording)' : 'Sources', title: inferred ? 'The answer did not cite its sources itself; these passages were matched to it afterwards.' : '' }), ...cites.map((c) => citationChip(m, c)));
    foot.append(h('span', { className: 'spacer' }));
    if (a.latency_sec) foot.append(h('span', { text: fmtSecs(a.latency_sec), title: a.queue_wait_s > 0.5 ? `including ${fmtSecs(a.queue_wait_s)} waiting in line` : 'Time to answer' }));
    foot.append(copyButton(m));
    card.append(foot);
    const trace = buildTrace(a);
    if (trace) card.append(trace);
    return card;
  }

  function buildTrace(a) {
    const trace = Array.isArray(a.trace) ? a.trace : [];
    if (!trace.length) return null;
    const total = trace.reduce((s, t) => s + (+t.seconds || 0), 0);
    const items = trace.map((t) => h('li', null,
      h('span', { className: 'trace-name', text: stepLabel(t.step) }),
      h('span', { className: 'trace-time', text: fmtSecs(+t.seconds || 0) }),
      describeStep(t) ? h('div', { className: 'trace-detail', text: describeStep(t) }) : null));
    const queries = Array.isArray(a.search_queries) && a.search_queries.length
      ? h('p', { className: 'trace-detail', text: `Searched for: ${a.search_queries.map((q) => `“${q}”`).join(', ')}` }) : null;
    return h('details', { className: 'trace' },
      h('summary', { text: `How this answer was found · ${plural(trace.length, 'step')} · ${fmtSecs(total)}` }),
      queries, h('ol', null, items));
  }
  function describeStep(t) {
    const d = t.detail || {};
    if (Array.isArray(d.pages)) return d.pages.length ? `Best passages on pages ${[...new Set(d.pages)].join(', ')}` : 'No passages found';
    if (Array.isArray(d.search_queries)) return `Queries: ${d.search_queries.join(' | ')}`;
    if (d.reason) return String(d.reason);
    if (typeof d.preview === 'string') return d.preview;
    if (d.output && typeof d.output === 'object' && 'grounded' in d.output) return d.output.grounded ? 'Every statement is supported by the passages' : 'Some statements are not supported';
    const s = JSON.stringify(d);
    return s && s !== '{}' ? (s.length > 220 ? s.slice(0, 219) + '…' : s) : '';
  }

  function copyButton(m) {
    const btn = h('button', { type: 'button', className: 'btn btn-ghost btn-sm', attrs: { 'aria-label': 'Copy answer' } }, icon('copy', 'icon icon-sm'), 'Copy');
    btn.addEventListener('click', async () => {
      const a = m.answer || {};
      let text = a.answer || '';
      if (Array.isArray(a.citations) && a.citations.length) {
        text += '\n\nSources:\n' + a.citations.map((c) => `[${c.marker}] ${state.docs.get(c.doc_id)?.title || c.doc_id}, page ${c.page}`).join('\n');
      }
      const ok = await copyText(text);
      btn.replaceChildren(icon(ok ? 'check' : 'alert', 'icon icon-sm'), ok ? 'Copied' : 'Copy failed');
      setTimeout(() => btn.replaceChildren(icon('copy', 'icon icon-sm'), 'Copy'), 1600);
    });
    return btn;
  }
  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); return true; } catch { /* blocked, e.g. in an iframe */ }
    const ta = h('textarea', { value: text, readOnly: true });
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.append(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch { ok = false; }
    ta.remove();
    return ok;
  }

  const ERROR_TITLES = {
    timeout: 'This is taking too long', busy: 'The server is busy', cancelled: 'Stopped',
    network: 'Connection problem', conflict: 'A document is not ready', missing: 'A document was removed', interrupted: 'Interrupted',
  };
  function buildError(m) {
    const e = m.error || {};
    return h('div', { className: 'card is-error' },
      h('div', { className: 'error-head' }, icon('alert'), ERROR_TITLES[e.code] || 'Something went wrong'),
      h('p', { text: e.detail || 'The answer could not be produced.' }),
      h('div', { className: 'answer-foot' }, retryButton(m)));
  }
  function retryButton(m) {
    return h('button', {
      type: 'button', className: 'btn btn-sm', disabled: !!state.pending, dataset: { role: 'retry-ask' },
      on: { click: () => runAsk(m, true) },
    }, icon('retry', 'icon icon-sm'), 'Try again');
  }

  // ================================================================ asking
  function historyPairs(beforeId) {
    const pairs = [];
    for (const m of state.messages) {
      if (m.id === beforeId) break;
      if (m.role === 'assistant' && m.status === 'done' && m.answer && !m.answer.error && m.answer.answer) pairs.push([m.question, m.answer.answer]);
    }
    return pairs.slice(-HISTORY_TURNS);
  }

  function trimAnswer(a) {
    const keep = ['question', 'answer', 'refused', 'citations', 'cited_pages', 'grounded', 'unsupported_claims',
      'search_queries', 'retrieval_rounds', 'answer_attempts', 'llm_calls', 'latency_sec', 'queue_wait_s', 'error'];
    const out = {};
    for (const k of keep) if (k in a) out[k] = a[k];
    if (a.partial && typeof a.partial === 'object') {
      out.partial = { doc_id: String(a.partial.doc_id || ''), pages_indexed: Number(a.partial.pages_indexed) || 0, pages_total: Number(a.partial.pages_total) || 0 };
    }
    out.trace = (Array.isArray(a.trace) ? a.trace : []).map((t) => {
      let detail = t.detail || {};
      try { if (JSON.stringify(detail).length > 1500) detail = { summary: describeStep(t) }; } catch { detail = {}; }
      return { step: t.step, seconds: t.seconds, detail };
    });
    return out;
  }

  function send(text) {
    const question = String(text || '').trim();
    if (!question || state.pending) return;
    if (question.length > config.maxQuestionChars) { toast(`Questions are limited to ${fmtInt(config.maxQuestionChars)} characters.`, 'error'); return; }
    if (!askableDocs().length) {
      toast(state.docs.size ? 'Please wait until a document has finished processing.' : 'Upload a PDF first.', 'error');
      return;
    }
    const user = { id: uid(), role: 'user', text: question, ts: Date.now() };
    const bot = {
      id: uid(), role: 'assistant', status: 'pending', question, docIds: scopeIds(), answerLang: answerLanguage(), ts: Date.now(),
      steps: [], phase: 'connecting', queuePos: 0, startedAt: Date.now(),
    };
    state.messages.push(user, bot);
    const input = $('composer-input');
    input.value = '';
    autosize();
    renderMessages();
    runAsk(bot, false);
  }

  async function runAsk(m, isRetry) {
    if (state.pending) return;
    if (isRetry) {
      // re-validate the scope: a selected document may have been deleted since
      if (m.docIds) m.docIds = m.docIds.filter((id) => isAskable(state.docs.get(id)));
      if (m.docIds && !m.docIds.length) m.docIds = null;
    }
    Object.assign(m, { status: 'pending', error: null, answer: null, steps: [], phase: 'connecting', queuePos: 0, startedAt: Date.now(), partial: '', streamDone: false });
    const controller = new AbortController();
    state.pending = { messageId: m.id, controller, reason: null };
    setComposerBusy(true);
    updateMessage(m);
    scrollToBottom(true);
    saveMessages();
    const clientTimeout = setTimeout(() => { state.pending.reason = 'timeout'; controller.abort(); }, (config.askTimeoutS * 2 + 60) * 1000);
    let lastRender = 0;
    let trailing = null;
    const rerender = (force) => {
      const now = Date.now();
      if (force || now - lastRender > 120) {
        lastRender = now;
        if (trailing) { clearTimeout(trailing); trailing = null; }
        updateMessage(m); scrollToBottom(false);
      } else if (!trailing) {
        // a throttled update must still happen once the stream goes quiet
        trailing = setTimeout(() => { trailing = null; if (m.status === 'pending') rerender(true); }, 130);
      }
    };
    try {
      const res = await fetch('api/ask/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({ question: m.question, doc_ids: m.docIds || null, history: historyPairs(m.id), answer_language: m.answerLang || null }),
        signal: controller.signal,
      });
      if (!res.ok) {
        let detail = `Request failed (HTTP ${res.status}).`;
        try { const d = await res.json(); if (typeof d.detail === 'string') detail = d.detail; } catch { /* not JSON */ }
        throw new ApiError(res.status, detail, res.status === 409 ? 'conflict' : res.status === 404 ? 'missing' : 'failed');
      }
      for await (const ev of readSSE(res.body)) {
        const d = ev.data || {};
        if (ev.event === 'queued') { m.phase = d.position > 0 ? 'queued' : 'thinking'; m.queuePos = d.position || 0; rerender(true); }
        else if (ev.event === 'started') { m.phase = 'thinking'; rerender(true); }
        else if (ev.event === 'step') {
          m.phase = 'thinking'; m.steps.push({ step: d.step, detail: d.detail, seconds: d.seconds });
          if (d.step === 'generate' && m.partial) m.streamDone = true; // text stopped: now verifying
          rerender(!!m.partial);
        }
        else if (ev.event === 'token') {
          m.phase = 'thinking';
          if (d.reset) { m.partial = ''; m.streamDone = false; }
          if (typeof d.text === 'string' && d.text) { m.streamDone = false; m.partial = m.partial ? m.partial + d.text : d.text.replace(/^\s+/, ''); }
          rerender(false);
        }
        else if (ev.event === 'answer') { m.answer = trimAnswer(d); m.status = 'done'; m.partial = ''; m.streamDone = false; }
        else if (ev.event === 'error') { m.status = 'error'; m.error = { code: d.code && d.code !== 'failed' ? d.code : d.status === 404 ? 'missing' : d.status === 409 ? 'conflict' : 'failed', detail: d.detail || 'Answering failed.' }; }
      }
      if (m.status === 'pending') {
        m.status = 'error';
        m.error = { code: 'network', detail: 'The connection closed before the answer arrived.' };
      }
    } catch (e) {
      m.status = 'error';
      if (e && e.name === 'AbortError') {
        m.error = state.pending && state.pending.reason === 'timeout'
          ? { code: 'timeout', detail: 'No answer arrived in time. The model may be overloaded.' }
          : { code: 'cancelled', detail: 'You stopped this answer.' };
      } else if (e instanceof ApiError) {
        m.error = { code: e.code || 'failed', detail: e.message };
      } else {
        m.error = { code: 'network', detail: 'The connection to the server was lost. Check that it is still running.' };
      }
    } finally {
      clearTimeout(clientTimeout);
      if (trailing) clearTimeout(trailing);
      m.partial = ''; m.streamDone = false;
      state.pending = null;
      setComposerBusy(false);
      updateMessage(m);
      scrollToBottom(false);
      saveMessages();
    }
  }

  function setComposerBusy(busy) {
    $('send').hidden = busy;
    $('stop').hidden = !busy;
    for (const b of document.querySelectorAll('[data-role="retry-ask"]')) b.disabled = busy;
  }

  function autosize() {
    const ta = $('composer-input');
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
    const n = ta.value.length;
    const cc = $('char-count');
    cc.textContent = n > config.maxQuestionChars * 0.75 ? `${fmtInt(n)} / ${fmtInt(config.maxQuestionChars)}` : '';
    cc.classList.toggle('is-over', n > config.maxQuestionChars);
  }

  // ================================================================ source viewer
  function isNarrow() { return window.matchMedia('(max-width: 819px)').matches; }
  function isMid() { return window.matchMedia('(min-width: 820px) and (max-width: 1199px)').matches; }

  function setView(view) {
    document.body.dataset.view = view;
    for (const t of document.querySelectorAll('.view-tab')) t.setAttribute('aria-current', String(t.dataset.viewTarget === view));
  }

  let lastChip = null;
  function openCitation(messageId, marker) {
    const m = state.messages.find((x) => x.id === messageId);
    const c = m && m.answer && (m.answer.citations || []).find((x) => x.marker === marker);
    if (!c) return;
    lastChip = document.activeElement;
    Object.assign(state.viewer, { docId: c.doc_id, page: c.page, citePage: c.page, chunkId: c.chunk_id, messageId, marker });
    for (const el of document.querySelectorAll('.chip.is-active')) el.classList.remove('is-active');
    for (const el of document.querySelectorAll(`.chip[data-msg="${CSS.escape(messageId)}"][data-marker="${marker}"]`)) el.classList.add('is-active');
    renderSources(messageId);
    showViewer();
    loadPage();
  }

  function showViewer() {
    if (isNarrow()) setView('viewer');
    document.body.classList.add('viewer-open');
    $('viewer-toolbar').hidden = false;
    $('page-empty').hidden = true;
    if (isMid() || isNarrow()) $('page-stage').focus({ preventScroll: true });
  }
  function closeViewer() {
    document.body.classList.remove('viewer-open');
    if (isNarrow()) setView('chat');
    if (lastChip && document.contains(lastChip)) lastChip.focus();
  }

  function renderSources(messageId) {
    const m = state.messages.find((x) => x.id === messageId);
    const cites = (m && m.answer && m.answer.citations) || [];
    $('sources').hidden = cites.length === 0;
    $('source-list').replaceChildren(...cites.map((c) => {
      const doc = state.docs.get(c.doc_id);
      const active = state.viewer.messageId === messageId && state.viewer.marker === c.marker;
      return h('li', null, h('button', {
        type: 'button', className: `source-item${active ? ' is-active' : ''}`,
        attrs: { 'aria-current': active ? 'true' : null },
        on: { click: () => openCitation(messageId, c.marker) },
      },
      h('span', { className: 'chip chip-static', text: `[${c.marker}]` }),
      h('span', { className: 'source-where', text: `${doc ? doc.title : 'Removed document'} · p. ${c.page}` }),
      h('span', { className: 'source-snippet', text: c.snippet || '' })));
    }));
  }

  function zoomLevel() { return ZOOM_LEVELS[state.viewer.zoomIndex]; }
  function stageWidth() { return Math.max(200, $('page-stage').clientWidth - 28); }

  function updateViewerChrome() {
    const v = state.viewer;
    const doc = state.docs.get(v.docId);
    const total = doc ? doc.num_pages : 0;
    $('viewer-title').textContent = doc ? doc.title : 'Source';
    $('viewer-sub').textContent = doc
      ? `Page ${fmtInt(v.page)}${total ? ` of ${fmtInt(total)}` : ''}${v.page === v.citePage && v.chunkId ? (v.highlighted ? ' · cited passage highlighted' : ' · cited passage is on this page') : ''}`
      : '';
    const input = $('page-input');
    if (document.activeElement !== input) input.value = String(v.page);
    input.max = total || '';
    $('page-total').textContent = total ? `/ ${fmtInt(total)}` : '';
    $('page-prev').disabled = v.page <= 1;
    $('page-next').disabled = !!total && v.page >= total;
    $('zoom-reset').textContent = `${Math.round(zoomLevel() * 100)}%`;
    $('zoom-out').disabled = v.zoomIndex === 0;
    $('zoom-in').disabled = v.zoomIndex === ZOOM_LEVELS.length - 1;
  }

  let pageToken = 0;
  function loadPage() {
    const v = state.viewer;
    if (!v.docId) return;
    v.highlighted = false; // set once the highlight boxes are known
    updateViewerChrome();
    const img = $('page-img');
    const width = Math.round(stageWidth() * zoomLevel());
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const renderZoom = Math.min(4, Math.max(0.5, Math.ceil(((width * dpr) / PAGE_PT_WIDTH) * 4) / 4));
    let url = `${docUrl(v.docId, `/pages/${v.page}.png`)}?zoom=${renderZoom}`;
    if (v.chunkId && v.page === v.citePage) url += `&chunk_id=${encodeURIComponent(v.chunkId)}`;
    const token = ++pageToken;
    // Where the cited passage sits on this page (0-1 fractions); fetched alongside the image.
    const boxes = v.chunkId && v.page === v.citePage
      ? api(docUrl(v.docId, `/pages/${v.page}/highlight?chunk_id=${encodeURIComponent(v.chunkId)}`))
        .then((r) => (Array.isArray(r && r.boxes) ? r.boxes : [])).catch(() => [])
      : Promise.resolve([]);
    $('page-error').hidden = true;
    $('page-loading').hidden = false;
    img.style.opacity = img.hidden ? '1' : '0.45';
    const probe = new Image();
    probe.onload = () => {
      if (token !== pageToken) return;
      img.src = url;
      img.alt = `Page ${v.page} of ${state.docs.get(v.docId)?.title || 'the document'}`;
      img.style.width = `${width}px`;
      img.style.opacity = '1';
      img.hidden = false;
      $('page-loading').hidden = true;
      boxes.then((b) => { if (token === pageToken) { v.highlighted = b.length > 0; updateViewerChrome(); scrollToHighlight(b[0], probe, width); } });
    };
    probe.onerror = async () => {
      if (token !== pageToken) return;
      $('page-loading').hidden = true;
      let msg = 'This page could not be rendered.';
      try { await api(url); } catch (e) { if (e.message) msg = e.message; }
      if (token === pageToken) showViewerError(msg);
    };
    probe.src = url;
  }
  // Bring the first highlighted box (0-1 page fractions) to the centre of the page stage.
  function scrollToHighlight(box, probe, width) {
    if (!box || !probe.naturalWidth) return;
    const stage = $('page-stage');
    const imgBox = $('page-img').getBoundingClientRect();
    const stageBox = stage.getBoundingClientRect();
    const height = width * (probe.naturalHeight / probe.naturalWidth);
    const x = imgBox.left - stageBox.left + stage.scrollLeft + ((box[0] + box[2]) / 2) * width;
    const y = imgBox.top - stageBox.top + stage.scrollTop + ((box[1] + box[3]) / 2) * height;
    const smooth = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    stage.scrollTo({ top: Math.max(0, y - stage.clientHeight / 2), left: Math.max(0, x - stage.clientWidth / 2), behavior: smooth ? 'smooth' : 'auto' });
  }
  function showViewerError(msg) {
    $('page-img').hidden = true;
    $('page-loading').hidden = true;
    const err = $('page-error');
    err.textContent = msg;
    err.hidden = false;
  }

  function goToPage(p) {
    const v = state.viewer;
    if (!v.docId) return;
    const total = state.docs.get(v.docId)?.num_pages || Infinity;
    const page = Math.min(Math.max(1, Math.round(p) || 1), total);
    if (page === v.page) { updateViewerChrome(); return; }
    v.page = page;
    $('page-stage').scrollTop = 0;
    loadPage();
  }
  function setZoom(idx) {
    state.viewer.zoomIndex = Math.min(ZOOM_LEVELS.length - 1, Math.max(0, idx));
    loadPage();
  }

  // ================================================================ health
  async function checkHealth() {
    try {
      const hInfo = await api('api/health');
      state.health = hInfo;
      setReachable(true);
      const s = hInfo.server || {};
      if (s.max_upload_mb) config.maxUploadMb = s.max_upload_mb;
      if (s.max_question_chars) { config.maxQuestionChars = s.max_question_chars; $('composer-input').maxLength = s.max_question_chars; }
      if (s.ask_timeout_s) config.askTimeoutS = s.ask_timeout_s;
      $('upload-limit').textContent = `Up to ${fmtBytes(config.maxUploadMb * MB)} per file. Large documents keep processing in the background.`;
      renderHealth();
    } catch (e) {
      if (e.code === 'network') setReachable(false);
    }
  }
  function setReachable(ok) {
    if (state.serverReachable === ok) return;
    state.serverReachable = ok;
    renderHealth();
  }
  function renderHealth() {
    const el = $('health');
    const text = el.querySelector('.health-text');
    const banner = $('banner');
    if (!state.serverReachable) {
      el.dataset.state = 'down';
      text.textContent = 'Server unreachable';
      el.title = 'The app cannot reach its server.';
      banner.textContent = 'Cannot reach the server. It may have stopped — reconnecting automatically…';
      banner.hidden = false;
      return;
    }
    const hInfo = state.health;
    if (!hInfo) return;
    const llmOk = hInfo.llm === 'ok';
    const device = hInfo.device === 'cuda' ? 'GPU' : hInfo.device === 'cpu' ? 'CPU' : (hInfo.device || '');
    const bits = [llmOk ? 'Model ready' : 'Model offline'];
    if (device) bits.push(device);
    const s = hInfo.server || {};
    if (s.asks_waiting) bits.push(`${s.asks_waiting} waiting`);
    else if (hInfo.queue) bits.push(`${plural(hInfo.queue, 'PDF')} queued`);
    el.dataset.state = llmOk ? (s.asks_waiting ? 'warn' : 'ok') : 'down';
    text.textContent = bits.join(' · ');
    el.title = hInfo.model ? `Model: ${hInfo.model}` : '';
    if (!llmOk) {
      banner.textContent = `The language model is not available (${hInfo.llm || 'unknown'}). Documents can still be uploaded, but answers will fail until it is back.`;
      banner.hidden = false;
    } else {
      banner.hidden = true;
    }
  }

  // ================================================================ wiring
  function wire() {
    const input = $('composer-input');
    $('composer').addEventListener('submit', (e) => { e.preventDefault(); send(input.value); });
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) { e.preventDefault(); send(input.value); }
    });
    input.addEventListener('input', autosize);
    $('stop').addEventListener('click', () => { if (state.pending) { state.pending.reason = 'user'; state.pending.controller.abort(); } });
    $('new-chat').addEventListener('click', () => {
      if (state.pending) { state.pending.reason = 'user'; state.pending.controller.abort(); }
      state.messages = [];
      saveMessages();
      renderMessages();
      $('sources').hidden = true;
      input.focus();
    });

    // uploads
    const fileInput = $('file-input');
    $('choose-files').addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => { addFiles(fileInput.files); fileInput.value = ''; });
    const dz = $('dropzone');
    const overlay = $('drop-overlay');
    let dragDepth = 0;
    const hasFiles = (e) => e.dataTransfer && [...(e.dataTransfer.types || [])].includes('Files');
    window.addEventListener('dragenter', (e) => { if (!hasFiles(e)) return; e.preventDefault(); dragDepth += 1; overlay.hidden = false; dz.classList.add('is-over'); });
    window.addEventListener('dragover', (e) => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; });
    window.addEventListener('dragleave', (e) => { if (!hasFiles(e)) return; dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) { overlay.hidden = true; dz.classList.remove('is-over'); } });
    window.addEventListener('drop', (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragDepth = 0;
      overlay.hidden = true;
      dz.classList.remove('is-over');
      addFiles(e.dataTransfer.files);
    });
    window.addEventListener('beforeunload', (e) => {
      if ([...state.uploads.values()].some((u) => u.status === 'uploading' || u.status === 'finishing')) { e.preventDefault(); e.returnValue = ''; }
    });

    $('select-toggle').addEventListener('click', () => { state.selected.clear(); saveSelected(); renderLibrary(); });
    initAnswerLanguage();

    // viewer
    $('page-prev').addEventListener('click', () => goToPage(state.viewer.page - 1));
    $('page-next').addEventListener('click', () => goToPage(state.viewer.page + 1));
    const pageInput = $('page-input');
    pageInput.addEventListener('change', () => goToPage(+pageInput.value));
    pageInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); goToPage(+pageInput.value); } });
    $('zoom-in').addEventListener('click', () => setZoom(state.viewer.zoomIndex + 1));
    $('zoom-out').addEventListener('click', () => setZoom(state.viewer.zoomIndex - 1));
    $('zoom-reset').addEventListener('click', () => setZoom(DEFAULT_ZOOM_INDEX));
    $('viewer-close').addEventListener('click', closeViewer);
    $('page-stage').addEventListener('keydown', (e) => {
      if (!state.viewer.docId) return;
      if (e.key === 'ArrowRight' || e.key === 'PageDown') { e.preventDefault(); goToPage(state.viewer.page + 1); }
      else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); goToPage(state.viewer.page - 1); }
      else if (e.key === '+' || e.key === '=') { e.preventDefault(); setZoom(state.viewer.zoomIndex + 1); }
      else if (e.key === '-') { e.preventDefault(); setZoom(state.viewer.zoomIndex - 1); }
    });
    let resizeTimer = null;
    let lastWidth = 0;
    new ResizeObserver(() => {
      const w = $('page-stage').clientWidth;
      if (!state.viewer.docId || Math.abs(w - lastWidth) < 24) return;
      lastWidth = w;
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(loadPage, 200);
    }).observe($('page-stage'));

    // tabs (phone layout)
    for (const t of document.querySelectorAll('.view-tab')) t.addEventListener('click', () => setView(t.dataset.viewTarget));

    document.addEventListener('keydown', (e) => {
      if (e.key !== 'Escape') return;
      if (state.confirmDelete) { const id = state.confirmDelete; setConfirm(null, id); return; }
      if (document.body.classList.contains('viewer-open') && (isMid() || isNarrow())) closeViewer();
    });

    document.addEventListener('visibilitychange', () => { if (!document.hidden) { loadDocs(); checkHealth(); } });
    setInterval(checkHealth, 15000);
  }

  function init() {
    setView('chat');
    wire();
    renderMessages();
    renderLibrary();
    autosize();
    scrollToBottom(true);
    checkHealth();
    loadDocs();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
