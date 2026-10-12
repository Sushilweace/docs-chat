"use strict";

/* Admin panel: read-only, paginated tables served by /admin/data/*.
   All text goes through textContent (never innerHTML), because it is user-controlled. */

const $ = (selector) => document.querySelector(selector);
const token = localStorage.getItem("token");

/* ---------- table definitions ---------- */
const fmtDate = (iso) => {
  if (!iso) return "";
  // SQLite returns naive timestamps that are really UTC.
  const date = new Date(/(Z|[+-]\d\d:?\d\d)$/i.test(iso) ? iso : `${iso}Z`);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
};

/** Long text, truncated to one line; click to expand/collapse. */
function clip(text) {
  const box = document.createElement("div");
  box.className = "clip";
  box.textContent = text ?? "";
  box.title = "Click to expand";
  box.addEventListener("click", () => box.classList.toggle("open"));
  return box;
}

/** A cell that drills into another table with a filter, e.g. conversation -> its messages. */
function drill(label, tab, filterKey, value, filterLabel) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "cell-link";
  button.textContent = label;
  button.title = `Show ${tab} for ${filterLabel}`;
  button.addEventListener("click", () => openTab(tab, { [filterKey]: { value, label: filterLabel } }));
  return button;
}

function badge(className, text) {
  const span = document.createElement("span");
  span.className = className;
  span.textContent = text;
  return span;
}

const TABS = {
  overview: { label: "Overview" },
  users: {
    label: "Users",
    search: "Search email…",
    columns: [
      { label: "ID", cls: "num", cell: (r) => r.id },
      { label: "Email", cell: (r) => r.email },
      { label: "Documents", cls: "num", cell: (r) => drill(r.documents, "documents", "user_id", r.id, `user #${r.id}`) },
      { label: "Chats", cls: "num", cell: (r) => drill(r.conversations, "conversations", "user_id", r.id, `user #${r.id}`) },
      { label: "Created", cls: "nowrap", cell: (r) => fmtDate(r.created_at) },
    ],
  },
  documents: {
    label: "Documents",
    search: "Search file name…",
    columns: [
      { label: "ID", cls: "num", cell: (r) => r.id },
      { label: "User", cell: (r) => `${r.user_email} (#${r.user_id})` },
      { label: "File", cell: (r) => clip(r.filename) },
      { label: "Status", cell: (r) => badge(`badge ${r.status}`, r.status) },
      { label: "Pages", cls: "num", cell: (r) => r.page_count },
      { label: "Chunks", cls: "num", cell: (r) => drill(r.chunk_count, "chunks", "document_id", r.id, `document #${r.id}`) },
      { label: "Error", cell: (r) => (r.error ? Object.assign(clip(r.error), { className: "clip err" }) : "") },
      { label: "Uploaded", cls: "nowrap", cell: (r) => fmtDate(r.created_at) },
    ],
  },
  conversations: {
    label: "Conversations",
    search: "Search title…",
    columns: [
      { label: "ID", cls: "num", cell: (r) => r.id },
      { label: "User", cell: (r) => `${r.user_email} (#${r.user_id})` },
      { label: "Title", cell: (r) => clip(r.title) },
      { label: "Messages", cls: "num", cell: (r) => drill(r.messages, "messages", "conversation_id", r.id, `chat #${r.id}`) },
      { label: "Started", cls: "nowrap", cell: (r) => fmtDate(r.created_at) },
    ],
  },
  messages: {
    label: "Messages",
    search: "Search message text…",
    columns: [
      { label: "ID", cls: "num", cell: (r) => r.id },
      { label: "Chat", cls: "num", cell: (r) => drill(`#${r.conversation_id}`, "messages", "conversation_id", r.conversation_id, `chat #${r.conversation_id}`) },
      { label: "User", cell: (r) => r.user_email },
      { label: "Role", cell: (r) => badge(`role ${r.role}`, r.role) },
      { label: "Content", cell: (r) => clip(r.content) },
      {
        label: "Cited",
        cell: (r) => (r.sources || []).map((s) => `${s.source} p${s.page}`).join(", "),
      },
      { label: "Sent", cls: "nowrap", cell: (r) => fmtDate(r.created_at) },
    ],
  },
  chunks: {
    label: "Chunks (vector DB)",
    search: "Search chunk text…",
    columns: [
      { label: "Chroma ID", cls: "mono", cell: (r) => r.id }, // "<document>:<page>:<chunk>"
      { label: "User", cls: "num", cell: (r) => r.user_id },
      { label: "Doc", cls: "num", cell: (r) => r.document_id },
      { label: "File", cell: (r) => r.source },
      { label: "Page", cls: "num", cell: (r) => r.page },
      { label: "#", cls: "num", cell: (r) => r.chunk_index },
      { label: "Chars", cls: "num", cell: (r) => r.chars },
      { label: "Text", cell: (r) => clip(r.text) },
    ],
  },
};

/* ---------- state ---------- */
const state = { tab: "overview", page: 1, pageSize: 20, q: "", filters: {}, pages: 1, total: 0, seq: 0 };

/* ---------- helpers ---------- */
async function getJson(path) {
  const res = await fetch(path, { headers: { Authorization: `Bearer ${token}` } });
  if (!res.ok) {
    const error = new Error(`Request failed (${res.status})`);
    error.status = res.status;
    throw error;
  }
  return res.json();
}

function showNotice(...parts) {
  const box = $("#notice");
  box.replaceChildren(...parts.map((p) => (typeof p === "string" ? document.createTextNode(p) : p)));
  box.classList.remove("hidden");
  $("#overview").classList.add("hidden");
  $("#table-view").classList.add("hidden");
}

function el(tag, text, attrs = {}) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  Object.assign(node, attrs);
  return node;
}

/* ---------- tabs ---------- */
function buildTabs() {
  const nav = $("#tabs");
  nav.replaceChildren();
  for (const [id, tab] of Object.entries(TABS)) {
    const button = el("button", tab.label, { type: "button" });
    button.dataset.tab = id;
    button.setAttribute("role", "tab");
    button.addEventListener("click", () => openTab(id));
    nav.appendChild(button);
  }
}

function openTab(id, filters = {}) {
  state.tab = id;
  state.page = 1;
  state.q = "";
  state.filters = filters;
  $("#search").value = "";
  history.replaceState(null, "", `#${id}`);
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === id));
  $("#notice").classList.add("hidden");

  if (id === "overview") {
    $("#table-view").classList.add("hidden");
    loadOverview();
  } else {
    $("#overview").classList.add("hidden");
    $("#table-view").classList.remove("hidden");
    $("#search").placeholder = TABS[id].search;
    renderHead();
    loadTable();
  }
}

/* ---------- overview ---------- */
async function loadOverview() {
  const box = $("#overview");
  try {
    const stats = await getJson("/admin/data/stats");
    box.replaceChildren();
    const labels = { users: "Users", documents: "Documents", conversations: "Conversations", messages: "Messages", chunks: "Chunks in vector DB" };
    for (const [key, label] of Object.entries(labels)) {
      const card = el("button", "", { type: "button", className: "card-stat" });
      card.append(el("span", String(stats[key]), { className: "num" }), el("span", label, { className: "label" }));
      card.addEventListener("click", () => openTab(key));
      box.appendChild(card);
    }
    box.classList.remove("hidden");
  } catch (err) {
    handleError(err);
  }
}

/* ---------- table ---------- */
function renderHead() {
  const row = el("tr");
  for (const col of TABS[state.tab].columns) row.appendChild(el("th", col.label));
  $("#grid thead").replaceChildren(row);
}

function renderFilters() {
  const box = $("#filters");
  box.replaceChildren();
  for (const [key, filter] of Object.entries(state.filters)) {
    const chip = el("span", filter.label, { className: "chip" });
    const remove = el("button", "✕", { type: "button", title: "Remove filter" });
    remove.setAttribute("aria-label", `Remove filter ${filter.label}`);
    remove.addEventListener("click", () => {
      delete state.filters[key];
      state.page = 1;
      loadTable();
    });
    chip.appendChild(remove);
    box.appendChild(chip);
  }
}

function renderRows(items) {
  const columns = TABS[state.tab].columns;
  const body = $("#grid tbody");
  body.replaceChildren();
  if (items.length === 0) {
    const row = el("tr");
    const cell = el("td", "No rows found.", { className: "empty", colSpan: columns.length });
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  for (const item of items) {
    const row = el("tr");
    for (const col of columns) {
      const cell = el("td");
      if (col.cls) cell.className = col.cls;
      const value = col.cell(item);
      if (value instanceof Node) cell.appendChild(value);
      else cell.textContent = value ?? "";
      row.appendChild(cell);
    }
    body.appendChild(row);
  }
}

function renderPager() {
  const { page, pageSize, total, pages } = state;
  const from = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const to = Math.min(page * pageSize, total);
  $("#range").textContent =
    total === 0 ? "No rows" : `${from}–${to} of ${total.toLocaleString()} row${total === 1 ? "" : "s"}`;
  $("#pages").textContent = String(pages);
  const input = $("#page-input");
  input.value = String(page);
  input.max = String(pages);
  $("#first").disabled = $("#prev").disabled = page <= 1;
  $("#next").disabled = $("#last").disabled = page >= pages;
}

async function loadTable() {
  const seq = ++state.seq; // ignore out-of-order responses (fast typing / clicking)
  const params = new URLSearchParams({ page: state.page, page_size: state.pageSize });
  if (state.q) params.set("q", state.q);
  for (const [key, filter] of Object.entries(state.filters)) params.set(key, filter.value);

  renderFilters();
  $("#table-view").classList.add("loading");
  try {
    const data = await getJson(`/admin/data/${state.tab}?${params}`);
    if (seq !== state.seq) return;
    if (data.total > 0 && state.page > data.pages) {
      state.page = data.pages; // e.g. rows were deleted since this page was opened
      return loadTable();
    }
    state.total = data.total;
    state.pages = data.pages;
    renderRows(data.items);
    renderPager();
  } catch (err) {
    if (seq === state.seq) handleError(err);
  } finally {
    if (seq === state.seq) $("#table-view").classList.remove("loading");
  }
}

function goToPage(page) {
  const clamped = Math.min(Math.max(1, Number(page) || 1), state.pages);
  if (clamped === state.page) return renderPager();
  state.page = clamped;
  loadTable();
}

/* ---------- errors & startup ---------- */
function handleError(err) {
  if (err.status === 401) {
    showNotice("Your session expired. ", el("a", "Log in again", { href: "/" }), ".");
  } else if (err.status === 403) {
    showNotice("You don't have admin access.");
  } else {
    showNotice(`Could not load data: ${err.message}`);
  }
}

async function boot() {
  if (!token) {
    showNotice("Please ", el("a", "log in", { href: "/" }), " first, then come back to this page.");
    return;
  }
  let me;
  try {
    me = await getJson("/auth/me");
  } catch (err) {
    if (err.status === 401) localStorage.removeItem("token");
    handleError(err);
    return;
  }
  $("#who").textContent = me.email;
  if (!me.is_admin) {
    showNotice(
      `${me.email} is not an admin. To allow it, add the email to `,
      el("code", "ADMIN_EMAILS"),
      " in the server's ",
      el("code", ".env"),
      " file (comma-separated) and restart the server.",
    );
    return;
  }

  buildTabs();
  $("#search").addEventListener("input", debounce((event) => {
    state.q = event.target.value.trim();
    state.page = 1;
    loadTable();
  }, 300));
  $("#page-size").addEventListener("change", (event) => {
    state.pageSize = Number(event.target.value);
    state.page = 1;
    loadTable();
  });
  $("#first").addEventListener("click", () => goToPage(1));
  $("#prev").addEventListener("click", () => goToPage(state.page - 1));
  $("#next").addEventListener("click", () => goToPage(state.page + 1));
  $("#last").addEventListener("click", () => goToPage(state.pages));
  $("#page-input").addEventListener("change", (event) => goToPage(event.target.value));

  const wanted = location.hash.slice(1);
  openTab(TABS[wanted] ? wanted : "overview");
}

function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

boot();
