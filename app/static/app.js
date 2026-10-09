"use strict";

/* ---------- helpers & state ---------- */
const $ = (selector) => document.querySelector(selector);

const state = {
  token: localStorage.getItem("token"),
  email: localStorage.getItem("email"),
  mode: "login", // "login" | "signup"
  conversationId: null,
  streaming: false,
  docs: [],
  pollTimer: null,
};

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function toast(message, isError = false) {
  const item = document.createElement("div");
  item.className = "toast-item" + (isError ? " error" : "");
  item.textContent = message;
  $("#toast").appendChild(item);
  setTimeout(() => item.remove(), 5000);
}

/** Turn a FastAPI error response into a human-readable message. */
async function errorMessage(res) {
  if (res.status === 429) return "Too many requests. Please wait a minute and try again.";
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail
        .map((e) => `${(e.loc || []).slice(1).join(".") || "input"}: ${e.msg}`)
        .join("; ");
    }
  } catch (_) {
    /* body was not JSON */
  }
  return `Request failed (${res.status})`;
}

/** fetch() wrapper: adds the bearer token and throws ApiError on non-2xx. */
async function api(path, { method = "GET", json, form } = {}) {
  const options = { method, headers: {} };
  if (state.token) options.headers.Authorization = `Bearer ${state.token}`;
  if (json !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(json);
  } else if (form) {
    options.body = form; // FormData / URLSearchParams set their own content type
  }

  let res;
  try {
    res = await fetch(path, options);
  } catch (_) {
    throw new ApiError(0, "Cannot reach the server. Is it running?");
  }
  if (res.status === 401 && state.token) {
    logout("Your session expired. Please log in again.");
    throw new ApiError(401, "Session expired");
  }
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res));
  return res;
}

/* ---------- views ---------- */
function showAuth() {
  $("#app-view").classList.add("hidden");
  $("#auth-view").classList.remove("hidden");
  $("#email").focus();
}

function showApp() {
  $("#auth-view").classList.add("hidden");
  $("#app-view").classList.remove("hidden");
  $("#user-email").textContent = state.email || "";
  $("#user-email").title = state.email || "";
  resetChat();
  loadDocuments();
}

function logout(message) {
  localStorage.removeItem("token");
  localStorage.removeItem("email");
  state.token = state.email = null;
  state.conversationId = null;
  clearTimeout(state.pollTimer);
  $("#messages").replaceChildren();
  $("#doc-list").replaceChildren();
  showAuth();
  if (message) toast(message, true);
}

/* ---------- auth ---------- */
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.mode === mode);
  });
  $("#auth-submit").textContent = mode === "login" ? "Log in" : "Create account";
  $("#password").autocomplete = mode === "login" ? "current-password" : "new-password";
  $("#auth-error").classList.add("hidden");
}

async function login(email, password) {
  const res = await api("/auth/login", { method: "POST", form: new URLSearchParams({ username: email, password }) });
  const { access_token } = await res.json();
  state.token = access_token;
  state.email = email;
  localStorage.setItem("token", access_token);
  localStorage.setItem("email", email);
}

async function handleAuthSubmit(event) {
  event.preventDefault();
  const email = $("#email").value.trim().toLowerCase();
  const password = $("#password").value;
  const errorBox = $("#auth-error");
  errorBox.classList.add("hidden");

  if (!email || !password) {
    errorBox.textContent = "Please enter your email and password.";
    errorBox.classList.remove("hidden");
    return;
  }

  const button = $("#auth-submit");
  button.disabled = true;
  try {
    if (state.mode === "signup") {
      await api("/auth/signup", { method: "POST", json: { email, password } });
    }
    await login(email, password);
    $("#password").value = "";
    showApp();
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove("hidden");
  } finally {
    button.disabled = false;
  }
}

/* ---------- documents ---------- */
function renderDocuments() {
  const list = $("#doc-list");
  list.replaceChildren();
  if (state.docs.length === 0) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No documents yet. Upload a PDF to get started.";
    list.appendChild(empty);
    return;
  }
  for (const doc of state.docs) {
    const item = document.createElement("li");
    item.className = "doc";

    const row = document.createElement("div");
    row.className = "row";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = doc.filename;
    name.title = doc.filename;
    const badge = document.createElement("span");
    badge.className = `badge ${doc.status}`;
    if (doc.status === "processing") {
      const spin = document.createElement("span");
      spin.className = "spinner";
      badge.appendChild(spin);
    }
    badge.appendChild(document.createTextNode(doc.status));
    row.append(name, badge);
    item.appendChild(row);

    const meta = document.createElement("div");
    meta.className = "muted";
    if (doc.status === "ready") {
      const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
      meta.textContent = `${plural(doc.page_count, "page")} · ${plural(doc.chunk_count, "chunk")}`;
    }
    else if (doc.status === "processing") meta.textContent = "Indexing… this can take a moment";
    item.appendChild(meta);

    if (doc.status === "failed" && doc.error) {
      const err = document.createElement("div");
      err.className = "err";
      err.textContent = doc.error;
      item.appendChild(err);
    }
    list.appendChild(item);
  }
}

async function loadDocuments() {
  clearTimeout(state.pollTimer);
  try {
    const res = await api("/documents");
    state.docs = await res.json();
  } catch (err) {
    if (err.status !== 401) toast(err.message, true);
    return;
  }
  renderDocuments();
  renderPlaceholderIfEmpty();
  // Keep refreshing while any document is still being ingested in the background.
  if (state.token && state.docs.some((d) => d.status === "processing")) {
    state.pollTimer = setTimeout(loadDocuments, 2000);
  }
}

async function uploadFiles(files) {
  for (const file of files) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      toast(`${file.name}: only PDF files are supported`, true);
      continue;
    }
    const form = new FormData();
    form.append("file", file);
    try {
      await api("/documents/upload", { method: "POST", form });
      toast(`Uploaded ${file.name}. Indexing in the background…`);
    } catch (err) {
      if (err.status !== 401) toast(`${file.name}: ${err.message}`, true);
    }
    await loadDocuments();
  }
}

/* ---------- chat rendering ---------- */
function messagesAtBottom() {
  const box = $("#messages");
  return box.scrollHeight - box.scrollTop - box.clientHeight < 80;
}

function scrollToBottom() {
  const box = $("#messages");
  box.scrollTop = box.scrollHeight;
}

function renderPlaceholderIfEmpty() {
  const box = $("#messages");
  if (box.querySelector(".msg")) return;
  const hasReady = state.docs.some((d) => d.status === "ready");
  const placeholder = box.querySelector(".placeholder") || document.createElement("div");
  placeholder.className = "placeholder";
  placeholder.replaceChildren();
  const icon = document.createElement("div");
  icon.className = "big";
  icon.textContent = hasReady ? "💬" : "📥";
  const text = document.createElement("p");
  text.textContent = hasReady
    ? "Ask a question about your documents. Answers cite the file and page they came from."
    : "Upload a PDF in the sidebar, wait until it shows READY, then ask your first question.";
  placeholder.append(icon, text);
  box.replaceChildren(placeholder);
}

/** Create a chat bubble; returns helpers to update it while streaming. */
function addMessage(role) {
  const box = $("#messages");
  box.querySelector(".placeholder")?.remove();

  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  wrapper.appendChild(bubble);
  box.appendChild(wrapper);
  scrollToBottom();

  let sources = [];
  let text = "";

  /** Render text, turning [n] markers into citation chips (tooltip: file + page). */
  function paint(streaming) {
    const stick = messagesAtBottom();
    bubble.replaceChildren();
    for (const part of text.split(/(\[\d+\])/)) {
      const match = part.match(/^\[(\d+)\]$/);
      const source = match && sources.find((s) => s.index === Number(match[1]));
      if (source) {
        const chip = document.createElement("sup");
        chip.className = "cite";
        chip.textContent = match[1];
        chip.title = `${source.source}, page ${source.page}`;
        bubble.appendChild(chip);
      } else if (part) {
        bubble.appendChild(document.createTextNode(part));
      }
    }
    bubble.classList.toggle("cursor", streaming);
    if (stick) scrollToBottom();
  }

  return {
    setText(value, streaming = false) {
      text = value;
      paint(streaming);
    },
    /** Remember the citations so [n] markers can become chips (not shown as a list). */
    setSources(value) {
      sources = value;
    },
    finish() {
      paint(false);
    },
    fail(message) {
      wrapper.classList.add("error");
      text = message;
      sources = [];
      paint(false);
    },
  };
}

function resetChat() {
  state.conversationId = null;
  $("#messages").replaceChildren();
  renderPlaceholderIfEmpty();
  $("#question").focus();
}

/* ---------- SSE over fetch (EventSource can't POST or send headers) ---------- */
async function readSSE(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      const dataLines = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length) onEvent(event, JSON.parse(dataLines.join("\n")));
    }
  }
}

function setBusy(busy) {
  state.streaming = busy;
  $("#send").disabled = busy;
  $("#question").disabled = busy;
  $("#new-chat").disabled = busy;
  if (!busy) $("#question").focus();
}

async function sendQuestion(question) {
  addMessage("user").setText(question);
  const reply = addMessage("assistant");
  reply.setText("", true);
  setBusy(true);

  let answer = "";
  let streamError = null;
  try {
    const payload = { question };
    if (state.conversationId) payload.conversation_id = state.conversationId;
    const res = await api("/chat", { method: "POST", json: payload });

    await readSSE(res, (event, data) => {
      if (event === "sources") reply.setSources(data);
      else if (event === "token") {
        answer += data.text;
        reply.setText(answer, true);
      } else if (event === "done") state.conversationId = data.conversation_id;
      else if (event === "error") streamError = data.detail;
    });

    if (streamError) reply.fail(streamError);
    else reply.finish();
  } catch (err) {
    if (err.status === 404 && state.conversationId) {
      state.conversationId = null;
      reply.fail("That conversation no longer exists. Please ask your question again to start a new one.");
    } else if (err.status !== 401) {
      reply.fail(err.message);
    }
  } finally {
    setBusy(false);
  }
}

/* ---------- wiring ---------- */
function autosizeTextarea() {
  const box = $("#question");
  box.style.height = "auto";
  const borders = box.offsetHeight - box.clientHeight; // box-sizing is border-box
  box.style.height = `${Math.min(box.scrollHeight + borders, 160)}px`;
}

function init() {
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.mode)));
  $("#auth-form").addEventListener("submit", handleAuthSubmit);
  $("#logout").addEventListener("click", () => logout());
  $("#new-chat").addEventListener("click", resetChat);

  $("#file-input").addEventListener("change", (event) => {
    const files = [...event.target.files];
    event.target.value = ""; // allow re-selecting the same file later
    if (files.length) uploadFiles(files);
  });
  const dropzone = $("#dropzone");
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      $("#file-input").click();
    }
  });
  ["dragenter", "dragover"].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (event) => uploadFiles([...event.dataTransfer.files]));

  const question = $("#question");
  question.addEventListener("input", autosizeTextarea);
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("#chat-form").requestSubmit();
    }
  });
  $("#chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = question.value.trim();
    if (!text || state.streaming) return;
    question.value = "";
    autosizeTextarea();
    sendQuestion(text);
  });

  if (state.token) showApp();
  else showAuth();
}

init();
