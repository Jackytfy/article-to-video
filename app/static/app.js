// Article -> Video minimal frontend.
// Talks to /jobs HTTP API + /jobs/{id}/events SSE stream.

const $ = (sel) => document.querySelector(sel);

const els = {
  form: $("#job-form"),
  submitBtn: $("#submit-btn"),
  activeSection: $("#active-section"),
  progressFill: $("#progress-fill"),
  progressPct: $("#progress-pct"),
  jobId: $("#job-id"),
  jobStatus: $("#job-status"),
  jobStage: $("#job-stage"),
  jobMood: $("#job-mood"),
  jobError: $("#job-error"),
  activeActions: $("#active-actions"),
  downloadMp4: $("#download-mp4"),
  downloadSrt: $("#download-srt"),
  refreshBtn: $("#refresh-btn"),
  jobsTbody: $("#jobs-tbody"),
};

let activeEventSource = null;

function setProgress(pct) {
  const clamped = Math.max(0, Math.min(1, pct));
  els.progressFill.style.width = `${(clamped * 100).toFixed(1)}%`;
  els.progressPct.textContent = `${Math.round(clamped * 100)}%`;
}

function paintJob(job) {
  els.jobId.textContent = job.job_id;
  els.jobStatus.textContent = job.status;
  els.jobStage.textContent = job.stage;
  els.jobMood.textContent = job.mood ?? "—";
  els.jobError.textContent = job.error ?? "—";
  setProgress(job.progress ?? 0);

  if (job.output_url) {
    els.activeActions.classList.remove("hidden");
    els.downloadMp4.href = job.output_url;
    els.downloadSrt.href = job.output_url.replace(/\/download$/, "/srt");
  }
}

function statusTag(status) {
  return `<span class="tag ${status}">${status}</span>`;
}

async function refreshList() {
  try {
    const resp = await fetch("/jobs");
    if (!resp.ok) throw new Error(resp.statusText);
    const { jobs } = await resp.json();

    if (!jobs.length) {
      els.jobsTbody.innerHTML = `<tr><td colspan="7" class="muted">无任务</td></tr>`;
      return;
    }

    els.jobsTbody.innerHTML = jobs
      .map((j) => {
        const created = new Date(j.created_at).toLocaleString();
        const dl = j.output_url
          ? `<a class="btn-link btn-secondary" href="${j.output_url}">下载</a>`
          : `<span class="muted">—</span>`;
        const live =
          j.status === "running"
            ? `<button class="btn-secondary" data-watch="${j.job_id}">订阅</button>`
            : "";
        return `<tr>
          <td><code>${j.job_id.slice(0, 8)}…</code></td>
          <td>${statusTag(j.status)}</td>
          <td>${j.stage}</td>
          <td>${Math.round((j.progress ?? 0) * 100)}%</td>
          <td>${j.mood ?? "—"}</td>
          <td>${created}</td>
          <td>${dl} ${live}</td>
        </tr>`;
      })
      .join("");

    els.jobsTbody.querySelectorAll("button[data-watch]").forEach((b) =>
      b.addEventListener("click", () => watchJob(b.dataset.watch))
    );
  } catch (err) {
    console.error("refreshList failed", err);
  }
}

function watchJob(jobId) {
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }

  els.activeSection.classList.remove("hidden");
  els.activeActions.classList.add("hidden");
  els.jobId.textContent = jobId;
  setProgress(0);

  const es = new EventSource(`/jobs/${jobId}/events`);
  activeEventSource = es;

  es.addEventListener("progress", (e) => {
    try {
      const job = JSON.parse(e.data);
      paintJob(job);
    } catch (err) {
      console.error("bad SSE payload", err);
    }
  });

  es.addEventListener("done", () => {
    es.close();
    activeEventSource = null;
    refreshList();
  });

  es.onerror = () => {
    // Network blip or terminal state — refresh state via REST.
    fetch(`/jobs/${jobId}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((job) => {
        if (job) paintJob(job);
      });
  };
}

function readForm() {
  const fd = new FormData(els.form);
  const translate_to = (fd.get("translate_to") || "").trim() || null;
  return {
    article: fd.get("article"),
    aspect_ratio: fd.get("aspect_ratio"),
    nlp_backend: fd.get("nlp_backend"),
    source_lang: fd.get("source_lang") || "zh",
    translate_to,
    voice_primary: fd.get("voice_primary"),
    voice_secondary: fd.get("voice_secondary") || null,
    bgm_enabled: fd.get("bgm_enabled") === "on",
    burn_subtitles: fd.get("burn_subtitles") === "on",
  };
}

els.form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  els.submitBtn.disabled = true;
  els.submitBtn.textContent = "提交中…";

  try {
    const resp = await fetch("/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`服务器拒绝: ${resp.status} ${txt}`);
    }
    const { job_id } = await resp.json();
    watchJob(job_id);
    refreshList();
  } catch (err) {
    alert(`提交失败：${err.message}`);
  } finally {
    els.submitBtn.disabled = false;
    els.submitBtn.textContent = "提交任务";
  }
});

els.refreshBtn.addEventListener("click", refreshList);

refreshList();
// 降低轮询频率：10 秒而不是 5 秒，减少服务器压力
setInterval(refreshList, 10000);

// 如果 SSE 连接活跃，暂停列表刷新（SSE 会更高效地推送更新）
if (activeEventSource) {
  clearInterval(refreshList);
}
