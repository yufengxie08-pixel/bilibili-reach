# -*- coding: utf-8 -*-
"""Local web app for Bilibili transcription MVP."""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from collections import deque
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from agent_reach.config import Config
from agent_reach.doctor import check_all
from agent_reach.transcribe import TranscribeError, transcribe

MAX_HISTORY = 50
APP_DIR = Config.CONFIG_DIR / "webapp"
HISTORY_FILE = APP_DIR / "history.json"


@dataclass
class TaskRecord:
    id: str
    source: str
    provider: str
    status: str
    transcript: str
    error: str
    created_at: float
    updated_at: float


class AppState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.history = self._load_history()

    def _load_history(self) -> deque[TaskRecord]:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        if not HISTORY_FILE.exists():
            return deque(maxlen=MAX_HISTORY)
        try:
            items = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return deque(maxlen=MAX_HISTORY)
        records = deque(maxlen=MAX_HISTORY)
        for item in items:
            records.append(TaskRecord(**item))
        return records

    def _save_history(self) -> None:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(
            json.dumps([asdict(item) for item in self.history], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_task(self, source: str, provider: str) -> TaskRecord:
        now = time.time()
        record = TaskRecord(
            id=str(int(now * 1000)),
            source=source,
            provider=provider,
            status="running",
            transcript="",
            error="",
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self.history.appendleft(record)
            self._save_history()
        return record

    def update_task(self, task_id: str, **fields: Any) -> TaskRecord | None:
        with self._lock:
            for record in self.history:
                if record.id == task_id:
                    for key, value in fields.items():
                        setattr(record, key, value)
                    record.updated_at = time.time()
                    self._save_history()
                    return record
        return None

    def list_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(item) for item in self.history]


STATE = AppState()


def run_transcription(task_id: str, source: str, provider: str) -> None:
    try:
        text = transcribe(source, provider=provider)
    except TranscribeError as exc:
        STATE.update_task(task_id, status="error", error=str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        STATE.update_task(task_id, status="error", error=f"Unexpected error: {exc}")
        return
    STATE.update_task(task_id, status="done", transcript=text, error="")


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode("utf-8"))


class AgentReachWebHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_index()
            return
        if parsed.path == "/api/history":
            _json_response(self, {"items": STATE.list_history()})
            return
        if parsed.path == "/api/status":
            cfg = Config()
            _json_response(
                self,
                {
                    "doctor": check_all(cfg),
                    "config": cfg.to_dict(),
                },
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/transcribe":
            payload = _read_json(self)
            source = (payload.get("source") or "").strip()
            provider = (payload.get("provider") or "auto").strip() or "auto"
            if not source:
                _json_response(self, {"error": "请先输入 B站链接或 BV 号"}, status=400)
                return
            record = STATE.add_task(source=source, provider=provider)
            thread = threading.Thread(
                target=run_transcription,
                args=(record.id, source, provider),
                daemon=True,
            )
            thread.start()
            _json_response(self, {"task": asdict(record)}, status=202)
            return

        if parsed.path == "/api/config":
            payload = _read_json(self)
            cfg = Config()
            updated = {}
            for field in ("groq_api_key", "openai_api_key"):
                value = payload.get(field)
                if value is not None:
                    value = str(value).strip()
                    if value:
                        cfg.set(field, value)
                    else:
                        cfg.delete(field)
                    updated[field] = bool(value)
            _json_response(self, {"ok": True, "updated": updated, "config": cfg.to_dict()})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _serve_index(self) -> None:
        html = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)


def run_web_app(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), AgentReachWebHandler)
    url = f"http://{host}:{port}"
    print(f"Bilibili Reach Web UI running at {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    server.serve_forever()


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bilibili Reach Local</title>
  <style>
    :root {
      --bg: #f5efe4;
      --paper: #fffaf2;
      --ink: #1e1a16;
      --muted: #73685c;
      --line: #d9cab6;
      --accent: #b0552b;
      --accent-2: #214d41;
      --ok: #216e39;
      --warn: #9a6700;
      --err: #b42318;
      --shadow: 0 18px 40px rgba(77, 53, 34, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Noto Serif SC", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(176,85,43,.12), transparent 30%),
        radial-gradient(circle at top right, rgba(33,77,65,.12), transparent 28%),
        linear-gradient(180deg, #f8f2e8 0%, var(--bg) 100%);
      min-height: 100vh;
    }
    .shell {
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }
    .hero {
      display: grid;
      grid-template-columns: 1.2fr .8fr;
      gap: 18px;
      margin-bottom: 18px;
    }
    .panel {
      background: color-mix(in srgb, var(--paper) 92%, white);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }
    .hero-copy {
      padding: 28px;
      position: relative;
      overflow: hidden;
    }
    .hero-copy::after {
      content: "";
      position: absolute;
      inset: auto -80px -100px auto;
      width: 220px;
      height: 220px;
      background: radial-gradient(circle, rgba(176,85,43,.16), transparent 65%);
      transform: rotate(18deg);
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(32px, 5vw, 54px);
      line-height: .98;
      letter-spacing: -.03em;
    }
    .lead {
      margin: 0;
      font-size: 16px;
      line-height: 1.7;
      color: var(--muted);
      max-width: 58ch;
    }
    .hero-stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      padding: 18px;
    }
    .stat {
      padding: 18px;
      border: 1px dashed var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,.55);
    }
    .stat strong {
      display: block;
      font-size: 28px;
      margin-bottom: 8px;
    }
    .tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 0 0 18px;
    }
    .tab {
      border: 1px solid var(--line);
      background: rgba(255,255,255,.72);
      color: var(--ink);
      padding: 10px 14px;
      border-radius: 999px;
      cursor: pointer;
      font-size: 14px;
    }
    .tab.active {
      background: var(--accent);
      color: #fff8f2;
      border-color: var(--accent);
    }
    .view { display: none; }
    .view.active { display: block; }
    .grid {
      display: grid;
      grid-template-columns: 1.1fr .9fr;
      gap: 18px;
    }
    .card {
      padding: 22px;
    }
    .title {
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .14em;
      color: var(--muted);
      margin-bottom: 10px;
    }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
      font: inherit;
      background: rgba(255,255,255,.8);
      color: var(--ink);
    }
    textarea {
      min-height: 138px;
      resize: vertical;
      line-height: 1.6;
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }
    button.primary, button.secondary {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      color: #fff8f2;
    }
    button.secondary {
      background: var(--accent-2);
      color: #eef8f1;
    }
    .result-box {
      min-height: 280px;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      background: rgba(255,255,255,.65);
      white-space: pre-wrap;
      line-height: 1.65;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(176,85,43,.1);
      color: var(--accent);
      font-size: 13px;
      margin-bottom: 12px;
    }
    .list {
      display: grid;
      gap: 12px;
    }
    .item {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255,255,255,.55);
    }
    .item h3 {
      margin: 0 0 8px;
      font-size: 16px;
      line-height: 1.4;
    }
    .meta {
      font-size: 13px;
      color: var(--muted);
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .status-ok { color: var(--ok); }
    .status-warn { color: var(--warn); }
    .status-error { color: var(--err); }
    .status-running { color: var(--accent); }
    .config-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    .small {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.6;
    }
    @media (max-width: 900px) {
      .hero, .grid, .config-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="panel hero-copy">
        <h1>Bilibili Reach Local</h1>
        <p class="lead">给你自己用的本地转写台。贴一个 B 站链接或 BV 号，直接拿文字稿。状态、密钥、历史都在一个页面里，不用再回终端来回切。</p>
      </div>
      <div class="panel hero-stats">
        <div class="stat">
          <strong>B站转写</strong>
          <span class="small">优先走现有字幕，失败时自动走音频 fallback。</span>
        </div>
        <div class="stat">
          <strong>本地运行</strong>
          <span class="small">你的 key 和历史只存在这台机器，不上云。</span>
        </div>
        <div class="stat">
          <strong>历史记录</strong>
          <span class="small">默认保留最近 50 条任务，方便回看与复用。</span>
        </div>
        <div class="stat">
          <strong>健康检查</strong>
          <span class="small">GUI 版 doctor，快速看哪个后端能用。</span>
        </div>
      </div>
    </section>

    <div class="tabs">
      <button class="tab active" data-view="transcribe">转写</button>
      <button class="tab" data-view="history">历史</button>
      <button class="tab" data-view="status">状态</button>
      <button class="tab" data-view="config">配置</button>
    </div>

    <section class="view active" id="view-transcribe">
      <div class="grid">
        <div class="panel card">
          <div class="title">输入链接</div>
          <textarea id="source" placeholder="粘贴 B站链接或直接输入 BV 号"></textarea>
          <div class="actions">
            <select id="provider">
              <option value="auto">自动选择（推荐）</option>
              <option value="groq">Groq</option>
              <option value="openai">OpenAI</option>
            </select>
            <button class="primary" id="runBtn">开始转写</button>
            <button class="secondary" id="copyBtn">复制结果</button>
            <button class="secondary" id="downloadBtn">下载 TXT</button>
          </div>
          <p class="small">提示：现在这个 MVP 优先服务你自己的 B站转写场景。后续要扩网页阅读、YouTube、RSS，也可以在这套壳上继续长。</p>
        </div>
        <div class="panel card">
          <div class="title">结果</div>
          <div class="pill" id="taskStatus">等待输入</div>
          <div class="result-box" id="resultBox">这里会显示转写结果、错误原因，或者当前任务进度。</div>
        </div>
      </div>
    </section>

    <section class="view" id="view-history">
      <div class="panel card">
        <div class="title">最近任务</div>
        <div class="list" id="historyList"></div>
      </div>
    </section>

    <section class="view" id="view-status">
      <div class="panel card">
        <div class="title">系统状态</div>
        <div class="list" id="statusList"></div>
      </div>
    </section>

    <section class="view" id="view-config">
      <div class="panel card">
        <div class="title">密钥配置</div>
        <div class="config-grid">
          <div>
            <label class="small" for="groqKey">Groq API Key</label>
            <input id="groqKey" placeholder="gsk_..." />
          </div>
          <div>
            <label class="small" for="openaiKey">OpenAI API Key</label>
            <input id="openaiKey" placeholder="sk-..." />
          </div>
        </div>
        <div class="actions">
          <button class="primary" id="saveConfigBtn">保存配置</button>
          <button class="secondary" id="refreshStatusBtn">刷新状态</button>
        </div>
        <p class="small" id="configHint">密钥只写入本机的 <code>~/.bilibili-reach/config.yaml</code>。</p>
      </div>
    </section>
  </div>

  <script>
    const tabs = [...document.querySelectorAll('.tab')];
    const views = [...document.querySelectorAll('.view')];
    const taskStatus = document.getElementById('taskStatus');
    const resultBox = document.getElementById('resultBox');
    const historyList = document.getElementById('historyList');
    const statusList = document.getElementById('statusList');
    const sourceInput = document.getElementById('source');
    const providerInput = document.getElementById('provider');
    const groqKeyInput = document.getElementById('groqKey');
    const openaiKeyInput = document.getElementById('openaiKey');
    let activeTaskId = null;
    let pollTimer = null;

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        views.forEach(v => v.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`view-${tab.dataset.view}`).classList.add('active');
        if (tab.dataset.view === 'history') loadHistory();
        if (tab.dataset.view === 'status') loadStatus();
      });
    });

    function setTaskPill(text, cls = 'status-running') {
      taskStatus.textContent = text;
      taskStatus.className = `pill ${cls}`;
    }

    async function loadHistory() {
      const res = await fetch('/api/history');
      const data = await res.json();
      historyList.innerHTML = data.items.length ? data.items.map(item => `
        <div class="item">
          <h3>${escapeHtml(item.source)}</h3>
          <div class="meta">
            <span class="status-${statusClass(item.status)}">${item.status}</span>
            <span>${new Date(item.updated_at * 1000).toLocaleString()}</span>
            <span>${escapeHtml(item.provider)}</span>
          </div>
          <p class="small">${escapeHtml(item.error || item.transcript.slice(0, 180) || '还没有内容')}</p>
        </div>
      `).join('') : '<p class="small">还没有历史任务。</p>';
    }

    async function loadStatus() {
      const res = await fetch('/api/status');
      const data = await res.json();
      const entries = Object.entries(data.doctor);
      statusList.innerHTML = entries.map(([key, item]) => `
        <div class="item">
          <h3>${escapeHtml(item.name)}</h3>
          <div class="meta">
            <span class="status-${statusClass(item.status)}">${item.status}</span>
            <span>${escapeHtml(item.active_backend || '—')}</span>
          </div>
          <p class="small">${escapeHtml(item.message)}</p>
        </div>
      `).join('');
    }

    function statusClass(status) {
      if (status === 'ok' || status === 'done') return 'ok';
      if (status === 'warn' || status === 'running') return 'warn';
      return 'error';
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }

    async function refreshActiveTask() {
      const res = await fetch('/api/history');
      const data = await res.json();
      const task = data.items.find(item => item.id === activeTaskId);
      if (!task) return;
      if (task.status === 'done') {
        setTaskPill('转写完成', 'status-ok');
        resultBox.textContent = task.transcript || '没有返回内容';
        clearInterval(pollTimer);
        loadHistory();
        return;
      }
      if (task.status === 'error') {
        setTaskPill('转写失败', 'status-error');
        resultBox.textContent = task.error || '未知错误';
        clearInterval(pollTimer);
        loadHistory();
        return;
      }
      setTaskPill('正在转写...', 'status-running');
      resultBox.textContent = '任务正在后台运行，请稍等。';
    }

    document.getElementById('runBtn').addEventListener('click', async () => {
      const source = sourceInput.value.trim();
      const provider = providerInput.value;
      if (!source) {
        setTaskPill('缺少链接', 'status-error');
        resultBox.textContent = '请先输入 B站链接或 BV 号。';
        return;
      }
      setTaskPill('提交任务中...', 'status-running');
      resultBox.textContent = '正在创建任务。';
      const res = await fetch('/api/transcribe', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({source, provider}),
      });
      const data = await res.json();
      if (!res.ok) {
        setTaskPill('提交失败', 'status-error');
        resultBox.textContent = data.error || '提交失败';
        return;
      }
      activeTaskId = data.task.id;
      setTaskPill('正在转写...', 'status-running');
      resultBox.textContent = '任务已开始，正在后台处理。';
      loadHistory();
      clearInterval(pollTimer);
      pollTimer = setInterval(refreshActiveTask, 1500);
      refreshActiveTask();
    });

    document.getElementById('copyBtn').addEventListener('click', async () => {
      await navigator.clipboard.writeText(resultBox.textContent || '');
      setTaskPill('结果已复制', 'status-ok');
    });

    document.getElementById('downloadBtn').addEventListener('click', () => {
      const blob = new Blob([resultBox.textContent || ''], {type: 'text/plain;charset=utf-8'});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'bilibili-reach-transcript.txt';
      link.click();
      URL.revokeObjectURL(link.href);
    });

    document.getElementById('saveConfigBtn').addEventListener('click', async () => {
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          groq_api_key: groqKeyInput.value,
          openai_api_key: openaiKeyInput.value,
        }),
      });
      const data = await res.json();
      document.getElementById('configHint').textContent = res.ok
        ? '配置已保存到本机。'
        : (data.error || '保存失败');
      loadStatus();
    });

    document.getElementById('refreshStatusBtn').addEventListener('click', loadStatus);

    loadHistory();
    loadStatus();
  </script>
</body>
</html>
"""
