// Vanilla JS demo client. Talks to POST /query and renders the answer,
// sources, per-stage trace waterfall, and this-query cost/latency.
// No framework, no build step - see ADR-0010 §Demo UI.

(() => {
  const form = document.getElementById("query-form");
  const questionEl = document.getElementById("question");
  const submitBtn = document.getElementById("submit-btn");
  const statusEl = document.getElementById("status");
  const errorBanner = document.getElementById("error-banner");
  const resultsEl = document.getElementById("results");
  const answerEl = document.getElementById("answer");
  const sourcesEl = document.getElementById("sources");
  const traceEl = document.getElementById("trace");
  const footerEl = document.getElementById("query-footer");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionEl.value.trim();
    if (!question) return;
    await runQuery(question);
  });

  async function runQuery(question) {
    setBusy(true);
    hideError();
    hideResults();
    resetResults();

    let sawFirstToken = false;
    try {
      // Streaming endpoint (ADR-0031): sources arrive first, then answer
      // tokens as they are generated, then a final metadata frame. This is
      // what drops the felt latency from ~4.4s to time-to-first-token.
      const res = await fetch("/query/stream", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question }),
      });

      // Pre-stream rejections (guardrail, empty, rate limit) come back as a
      // normal JSON error, not SSE - handle them exactly as before.
      if (!res.ok || !res.body) {
        await handleErrorResponse(res);
        return;
      }

      resultsEl.hidden = false;
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let sep;
        while ((sep = buffer.indexOf("\n\n")) !== -1) {
          const frame = parseFrame(buffer.slice(0, sep));
          buffer = buffer.slice(sep + 2);
          if (!frame) continue;

          if (frame.event === "sources") {
            renderSources(frame.data.sources || []);
          } else if (frame.event === "token") {
            if (!sawFirstToken) {
              sawFirstToken = true;
              statusEl.textContent = "Streaming…";
            }
            answerEl.textContent += frame.data.text || "";
          } else if (frame.event === "done") {
            renderTrace(frame.data.trace || []);
            renderFooter(frame.data);
          } else if (frame.event === "error") {
            showError({
              kind: "error",
              title: "Streaming failed",
              message: frame.data.message || "Something went wrong. Please try again.",
            });
          }
        }
      }
    } catch (err) {
      showError({
        kind: "error",
        title: "Network error",
        message:
          "Could not reach the demo. The service may be scaling up from zero - try again in a few seconds.",
      });
    } finally {
      setBusy(false);
    }
  }

  // Parse one SSE frame ("event: X\ndata: {...}") into {event, data}, or null.
  function parseFrame(frame) {
    let event = null;
    let data = null;
    for (const line of frame.split("\n")) {
      if (line.startsWith("event: ")) {
        event = line.slice(7);
      } else if (line.startsWith("data: ")) {
        try {
          data = JSON.parse(line.slice(6));
        } catch {
          data = null;
        }
      }
    }
    if (event === null || data === null) return null;
    return { event, data };
  }

  async function handleErrorResponse(res) {
    let body = null;
    try {
      body = await res.json();
    } catch {
      // Fall through to generic
    }
    const errorType = body?.error_type;

    if (errorType === "demo_daily_limit_reached") {
      showError({
        kind: "warning",
        title: "Daily demo limit reached",
        message:
          "This demo is capped at 200 questions per day (see About). It will reset at 00:00 UTC. Thanks for stopping by.",
      });
    } else if (errorType === "demo_disabled") {
      showError({
        kind: "warning",
        title: "Demo temporarily disabled",
        message: "The owner has paused the demo. Please check back later.",
      });
    } else if (res.status === 429) {
      showError({
        kind: "warning",
        title: "Slow down a moment",
        message:
          "Rate limit reached for your IP (10/hour, 3/minute burst). Try again in a minute.",
      });
    } else if (errorType === "guardrail_rejection") {
      showError({
        kind: "warning",
        title: "Input rejected",
        message:
          body?.detail ||
          "The input guardrail rejected this question. Try rephrasing it as a documentation question.",
      });
    } else if (errorType === "not_ready") {
      showError({
        kind: "error",
        title: "Service not ready",
        message:
          "A dependency is unavailable right now. The daily eval run may be regenerating the index - try again shortly.",
      });
    } else {
      showError({
        kind: "error",
        title: `Unexpected response (${res.status})`,
        message: body?.message || "Something went wrong. Please try again.",
      });
    }
  }

  function resetResults() {
    answerEl.textContent = "";
    sourcesEl.innerHTML = "";
    traceEl.innerHTML = "";
    footerEl.innerHTML = "";
  }

  function renderSources(sources) {
    sourcesEl.innerHTML = "";
    if (Array.isArray(sources) && sources.length > 0) {
      for (const src of sources) {
        const li = document.createElement("li");
        const file = document.createElement("code");
        file.textContent = src.source_file;
        li.appendChild(file);

        if (Array.isArray(src.heading_path) && src.heading_path.length > 0) {
          const path = document.createElement("span");
          path.className = "heading-path";
          path.textContent = "  ·  " + src.heading_path.join(" › ");
          li.appendChild(path);
        }
        sourcesEl.appendChild(li);
      }
    } else {
      const li = document.createElement("li");
      li.textContent = "(no sources returned)";
      sourcesEl.appendChild(li);
    }
  }

  function renderTrace(spans) {
    traceEl.innerHTML = "";
    if (spans.length === 0) {
      traceEl.textContent = "(no spans recorded)";
      return;
    }
    const maxDuration = Math.max(...spans.map((s) => s.duration_ms || 0), 1);

    for (const s of spans) {
      const row = document.createElement("div");
      row.className = "trace-row";

      const name = document.createElement("div");
      name.className = "trace-name";
      name.textContent = s.name;

      const track = document.createElement("div");
      track.className = "trace-bar-track";
      const fill = document.createElement("div");
      fill.className = "trace-bar-fill";
      const pct = Math.max(1, ((s.duration_ms || 0) / maxDuration) * 100);
      fill.style.width = pct + "%";
      track.appendChild(fill);

      const dur = document.createElement("div");
      dur.className = "trace-duration";
      dur.textContent = formatMs(s.duration_ms || 0);

      row.appendChild(name);
      row.appendChild(track);
      row.appendChild(dur);
      traceEl.appendChild(row);
    }
  }

  function renderFooter(body) {
    footerEl.innerHTML = "";
    const parts = [
      ["First token", body.ttft_ms != null ? formatMs(body.ttft_ms) : "-"],
      ["Cost", body.cost_usd != null ? formatCost(body.cost_usd) : "-"],
      ["Latency", body.latency_ms != null ? formatMs(body.latency_ms) : "-"],
      ["Spans", String((body.trace || []).length)],
    ];
    for (const [label, value] of parts) {
      const wrap = document.createElement("span");
      const l = document.createElement("span");
      l.className = "metric-label";
      l.textContent = label + ": ";
      const v = document.createElement("span");
      v.className = "metric-value";
      v.textContent = value;
      wrap.appendChild(l);
      wrap.appendChild(v);
      footerEl.appendChild(wrap);
    }
  }

  function setBusy(busy) {
    submitBtn.disabled = busy;
    submitBtn.textContent = busy ? "Asking…" : "Ask";
    statusEl.textContent = busy ? "Thinking…" : "";
  }

  function showError({ kind, title, message }) {
    errorBanner.className = "error-banner" + (kind === "error" ? " error" : "");
    errorBanner.innerHTML = "";
    const t = document.createElement("strong");
    t.textContent = title;
    const m = document.createTextNode(message);
    errorBanner.appendChild(t);
    errorBanner.appendChild(m);
    errorBanner.hidden = false;
  }

  function hideError() {
    errorBanner.hidden = true;
    errorBanner.innerHTML = "";
  }

  function hideResults() {
    resultsEl.hidden = true;
  }

  function formatMs(ms) {
    if (ms < 10) return ms.toFixed(2) + " ms";
    if (ms < 1000) return Math.round(ms) + " ms";
    return (ms / 1000).toFixed(2) + " s";
  }

  function formatCost(usd) {
    if (usd === 0) return "$0.00";
    if (usd < 0.001) return "< $0.001";
    return "$" + usd.toFixed(4);
  }
})();
