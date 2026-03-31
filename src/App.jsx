import { useState, useRef, useEffect, useMemo } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid,
} from "recharts";

// ─────────────────────────────────────────────
//  OLLAMA API  (local — tinyllama / mistral)
//  ollama serve   (with OLLAMA_ORIGINS=*)
//  ollama pull tinyllama
// ─────────────────────────────────────────────
const OLLAMA_BASE  = "http://localhost:11434";
const OLLAMA_MODEL = "tinyllama";

async function ollama(prompt, system = "", maxTokens = 2048) {
  const messages = [];
  if (system) messages.push({ role: "system", content: system });
  messages.push({ role: "user", content: prompt });
  const res = await fetch(`${OLLAMA_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      messages,
      stream: false,
      options: { num_predict: maxTokens, temperature: 0.6 },
    }),
  });
  if (!res.ok) throw new Error(`Ollama HTTP ${res.status}`);
  const data = await res.json();
  return data?.message?.content || "";
}

async function checkOllama() {
  try {
    const res = await fetch(`${OLLAMA_BASE}/api/tags`);
    if (!res.ok) return { ok: false, models: [] };
    const data = await res.json();
    return { ok: true, models: (data?.models || []).map(m => m.name) };
  } catch { return { ok: false, models: [] }; }
}

// ─────────────────────────────────────────────
//  JSON PARSER — robust, handles tinyllama slop
// ─────────────────────────────────────────────
function parseJSON(raw) {
  if (!raw) return null;
  // strip markdown fences
  let s = raw.replace(/```json[\s\S]*?```/gi, m => m.slice(7, -3))
             .replace(/```[\s\S]*?```/g, m => m.slice(3, -3))
             .trim();
  // direct parse
  try { return JSON.parse(s); } catch {}
  // extract first array
  const arr = s.match(/\[[\s\S]*\]/);
  if (arr) { try { return JSON.parse(arr[0]); } catch {} }
  // extract first object
  const obj = s.match(/\{[\s\S]*\}/);
  if (obj) { try { return JSON.parse(obj[0]); } catch {} }
  return null;
}

// ─────────────────────────────────────────────
//  RAG — chunking
// ─────────────────────────────────────────────
function chunkText(text, chunkSize = 400, overlap = 80) {
  const words = text.split(/\s+/).filter(Boolean);
  const chunks = [];
  let i = 0;
  while (i < words.length) {
    chunks.push({ id: chunks.length, text: words.slice(i, i + chunkSize).join(" "), wordStart: i });
    i += chunkSize - overlap;
  }
  return chunks;
}

function buildVocab(chunks) {
  const vocab = {};
  let idx = 0;
  chunks.forEach(c =>
    c.text.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).forEach(w => {
      if (w.length > 2 && !vocab[w]) vocab[w] = idx++;
    })
  );
  return vocab;
}

function buildIdf(chunks, vocab) {
  const df = {};
  chunks.forEach(c => {
    new Set(c.text.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(w => w.length > 2))
      .forEach(w => { df[w] = (df[w] || 0) + 1; });
  });
  const idf = {};
  Object.keys(vocab).forEach(w => {
    idf[w] = Math.log((chunks.length + 1) / ((df[w] || 0) + 1)) + 1;
  });
  return idf;
}

function tfIdfVector(text, vocab, idfMap) {
  const words = text.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(w => w.length > 2);
  const tf = {};
  words.forEach(w => { tf[w] = (tf[w] || 0) + 1; });
  const vec = new Float32Array(Object.keys(vocab).length);
  Object.entries(tf).forEach(([w, c]) => {
    if (vocab[w] !== undefined) vec[vocab[w]] = (c / words.length) * (idfMap[w] || 1);
  });
  return vec;
}

function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

// ─────────────────────────────────────────────
//  IN-MEMORY VECTOR STORE
// ─────────────────────────────────────────────
class VectorStore {
  constructor() { this._col = []; this._vocab = null; this._idf = null; }
  insertMany(chunks, vocab, idf) {
    this._vocab = vocab; this._idf = idf;
    this._col = chunks.map(c => ({
      id: `chunk_${c.id}`, text: c.text,
      vector: tfIdfVector(c.text, vocab, idf),
    }));
  }
  search(query, topK = 5) {
    if (!this._col.length || !this._vocab) return [];
    const qv = tfIdfVector(query, this._vocab, this._idf);
    return this._col
      .map(d => ({ ...d, score: cosineSim(qv, d.vector) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
  get size() { return this._col.length; }
}
const vs = new VectorStore();

// ─────────────────────────────────────────────
//  SESSION STORE (sessionStorage as MongoDB stub)
// ─────────────────────────────────────────────
const MongoDB = {
  sessions: JSON.parse(sessionStorage.getItem("qai_sessions") || "[]"),
  save(doc) {
    this.sessions = [doc, ...this.sessions].slice(0, 20);
    sessionStorage.setItem("qai_sessions", JSON.stringify(this.sessions));
    console.log("[MongoDB] insertOne:", JSON.stringify(doc).slice(0, 120));
  },
};

let LEADERBOARD = JSON.parse(sessionStorage.getItem("qai_lb") || "[]");
const saveLB = e => {
  LEADERBOARD = [e, ...LEADERBOARD].slice(0, 10).sort((a, b) => b.pct - a.pct);
  sessionStorage.setItem("qai_lb", JSON.stringify(LEADERBOARD));
};

// ─────────────────────────────────────────────
//  HELPERS
// ─────────────────────────────────────────────
const LETTERS   = ["A", "B", "C", "D"];
const fmtTime   = s => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
const DIFFS     = ["easy", "medium", "hard"];
const normDiff  = d => DIFFS.includes((d || "").toLowerCase()) ? d.toLowerCase() : "medium";
const diffColor = { easy: "#34D399", medium: "#FBBF24", hard: "#F87171" };

const getRank = pct => {
  if (pct >= 95) return { rank: "S+", label: "Legendary",   color: "#FFD700", glow: "rgba(255,215,0,0.4)",   emoji: "👑" };
  if (pct >= 90) return { rank: "S",  label: "Outstanding", color: "#E879F9", glow: "rgba(232,121,249,0.4)", emoji: "⭐" };
  if (pct >= 80) return { rank: "A",  label: "Excellent",   color: "#34D399", glow: "rgba(52,211,153,0.4)",  emoji: "🏆" };
  if (pct >= 70) return { rank: "B",  label: "Good",        color: "#38BDF8", glow: "rgba(56,189,248,0.4)",  emoji: "🎯" };
  if (pct >= 60) return { rank: "C",  label: "Average",     color: "#FBBF24", glow: "rgba(251,191,36,0.4)",  emoji: "📚" };
  if (pct >= 40) return { rank: "D",  label: "Needs Work",  color: "#FB923C", glow: "rgba(251,146,60,0.4)",  emoji: "📖" };
  return               { rank: "F",  label: "Keep Trying", color: "#F87171", glow: "rgba(248,113,113,0.4)", emoji: "💪" };
};

const readFileAsText = f => new Promise((res, rej) => {
  const r = new FileReader();
  r.onload = () => res(r.result);
  r.onerror = rej;
  r.readAsText(f);
});

// ─────────────────────────────────────────────
//  TINYLLAMA-OPTIMISED PROMPTS
//  tinyllama is small — keep prompts SHORT & STRUCTURED
// ─────────────────────────────────────────────
function buildQuizPrompt(context, numQ, qType, difficulty) {
  const typeInstr =
    qType === "mcq"   ? `Generate ${numQ} MCQ questions ONLY.` :
    qType === "short" ? `Generate ${numQ} short-answer questions ONLY.` :
                        `Generate ${Math.ceil(numQ / 2)} MCQ and ${Math.floor(numQ / 2)} short-answer questions.`;

  const diffInstr = difficulty === "mixed"
    ? "Mix easy, medium, hard difficulties evenly."
    : `All questions should be ${difficulty} difficulty.`;

  return `You are a quiz generator. Output ONLY a valid JSON array. No explanation. No markdown.

Each MCQ item: {"id":1,"type":"mcq","topic":"<topic>","difficulty":"easy","question":"<q>","options":["A","B","C","D"],"answer":"<exact option text>","explanation":"<1 sentence>"}
Each short item: {"id":1,"type":"short","topic":"<topic>","difficulty":"medium","question":"<q>","answer":"<model answer>","keywords":["kw1","kw2"],"hint":"<hint>"}

Rules:
- ${typeInstr}
- ${diffInstr}
- Base questions ONLY on the context below.
- "answer" for MCQ must be the EXACT text of one option.
- Output a JSON array and nothing else.

Context:
${context.slice(0, 3000)}

JSON array:`;
}

// ─────────────────────────────────────────────
//  STYLES
// ─────────────────────────────────────────────
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Playfair+Display:ital,wght@0,700;0,900;1,600;1,700&display=swap');

:root {
  --ink:#03020A;--ink1:#080714;--ink2:#0E0C1E;--ink3:#151228;--ink4:#1C1835;
  --glass:rgba(255,255,255,0.028);--glass2:rgba(255,255,255,0.055);--glass3:rgba(255,255,255,0.09);
  --rim:rgba(255,255,255,0.06);--rim2:rgba(255,255,255,0.11);
  --violet:#7C3AED;--violet2:#A78BFA;--violet3:#C4B5FD;
  --cyan:#06B6D4;--cyan2:#67E8F9;--rose:#FB7185;--gold:#F59E0B;--emerald:#10B981;--sky:#38BDF8;
  --text:#EEE9FF;--text2:#8B82B0;--text3:#3D3860;
  --r:18px;--r2:12px;--r3:8px;
  --brand:#F97316;--brand2:#FED7AA;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--ink);color:var(--text);font-family:'Syne',system-ui,sans-serif;-webkit-font-smoothing:antialiased;min-height:100vh;overflow-x:hidden}
::selection{background:rgba(124,58,237,0.35);color:var(--violet3)}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-thumb{background:rgba(124,58,237,0.25);border-radius:4px}

/* BG */
.bg-canvas{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.nebula{position:absolute;border-radius:50%;filter:blur(120px)}
.nb1{width:900px;height:900px;top:-300px;left:-200px;background:radial-gradient(circle,rgba(124,58,237,0.09) 0%,transparent 70%);animation:nb 25s ease-in-out infinite alternate}
.nb2{width:700px;height:700px;top:30%;right:-200px;background:radial-gradient(circle,rgba(6,182,212,0.06) 0%,transparent 70%);animation:nb 30s ease-in-out infinite alternate-reverse}
.nb3{width:600px;height:600px;bottom:-100px;left:20%;background:radial-gradient(circle,rgba(249,115,22,0.04) 0%,transparent 70%);animation:nb 22s ease-in-out infinite alternate}
@keyframes nb{0%{transform:translate(0,0)scale(1)}100%{transform:translate(60px,30px)scale(1.06)}}
.stars{position:absolute;inset:0;background-image:radial-gradient(1px 1px at 10% 20%,rgba(255,255,255,0.4) 0%,transparent 100%),radial-gradient(1px 1px at 30% 70%,rgba(255,255,255,0.3) 0%,transparent 100%),radial-gradient(1.5px 1.5px at 60% 15%,rgba(200,180,255,0.5) 0%,transparent 100%),radial-gradient(1px 1px at 80% 55%,rgba(255,255,255,0.25) 0%,transparent 100%),radial-gradient(1.5px 1.5px at 92% 30%,rgba(255,255,255,0.35) 0%,transparent 100%);animation:twinkle 8s ease-in-out infinite alternate}
@keyframes twinkle{0%{opacity:.6}100%{opacity:1}}
.grid-bg{position:absolute;inset:0;background-image:linear-gradient(rgba(124,58,237,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(124,58,237,0.03) 1px,transparent 1px);background-size:80px 80px;mask-image:radial-gradient(ellipse 80% 80% at 50% 50%,black 0%,transparent 100%)}
.scanlines{position:absolute;inset:0;pointer-events:none;z-index:1;background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.03) 3px,rgba(0,0,0,0.03) 4px)}

/* ANIMATIONS */
@keyframes slide-up{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:translateY(0)}}
@keyframes fade-in{from{opacity:0}to{opacity:1}}
@keyframes pop{0%{transform:scale(0.5);opacity:0}70%{transform:scale(1.06)}100%{transform:scale(1);opacity:1}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes spin-rev{to{transform:rotate(-360deg)}}
@keyframes pulse-glow{0%,100%{opacity:1}50%{opacity:0.4}}
@keyframes shimmer{0%{background-position:200% center}100%{background-position:-200% center}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes halo{to{transform:rotate(360deg)}}
@keyframes bar-fill{from{width:0}}

.au{animation:slide-up .6s cubic-bezier(.16,1,.3,1) both}
.au1{animation:slide-up .6s cubic-bezier(.16,1,.3,1) .1s both}
.au2{animation:slide-up .6s cubic-bezier(.16,1,.3,1) .2s both}
.au3{animation:slide-up .6s cubic-bezier(.16,1,.3,1) .3s both}
.pop{animation:pop .7s cubic-bezier(.34,1.56,.64,1) both}
.fi{animation:fade-in .5s ease both}

.page{position:relative;z-index:1;min-height:100vh}

/* NAV */
.nav{display:flex;align-items:center;gap:14px;padding:14px 40px;background:rgba(3,2,10,.75);backdrop-filter:blur(32px) saturate(1.5);border-bottom:1px solid var(--rim);position:sticky;top:0;z-index:200}
.nav-mark{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#F97316 0%,#7C3AED 100%);display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;box-shadow:0 0 0 1px rgba(249,115,22,.4),0 0 24px rgba(249,115,22,.3)}
.nav-wordmark{font-size:17px;font-weight:800;letter-spacing:-.8px}
.nav-wordmark em{font-style:normal;background:linear-gradient(90deg,#F97316,#A78BFA);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.nav-tag{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.12em;padding:3px 10px;border-radius:30px;text-transform:uppercase;background:rgba(249,115,22,.1);border:1px solid rgba(249,115,22,.3);color:var(--brand2)}
.nav-r{margin-left:auto;display:flex;align-items:center;gap:10px}
.nav-user{font-family:'DM Mono',monospace;font-size:11px;color:var(--cyan2);padding:5px 14px;border-radius:30px;background:rgba(6,182,212,.07);border:1px solid rgba(6,182,212,.2)}
.nav-steps{display:flex;align-items:center;gap:2px}
.nstep{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.08em;text-transform:uppercase;padding:5px 12px;border-radius:30px;border:1px solid transparent;color:var(--text3);transition:all .3s}
.nstep::after{content:'›';margin-left:5px;opacity:.3}.nstep:last-child::after{display:none}
.nstep.on{color:var(--brand2);border-color:rgba(249,115,22,.3);background:rgba(249,115,22,.1)}
.nstep.done{color:var(--emerald);border-color:rgba(16,185,129,.25)}

/* STATUS BADGE */
.o-status{display:flex;align-items:center;gap:8px;padding:5px 14px;border-radius:30px;font-family:'DM Mono',monospace;font-size:10px;border:1px solid;transition:all .3s}
.o-status.connected{color:#4ADE80;border-color:rgba(74,222,128,.3);background:rgba(74,222,128,.06)}
.o-status.connected .os-dot{background:#4ADE80;box-shadow:0 0 8px #4ADE80;animation:pulse-glow 2s ease infinite}
.o-status.disconnected{color:#F87171;border-color:rgba(248,113,113,.3);background:rgba(248,113,113,.06)}
.o-status.disconnected .os-dot{background:#F87171}
.o-status.checking{color:var(--text3);border-color:var(--rim)}
.o-status.checking .os-dot{background:var(--gold);animation:pulse-glow 1s ease infinite}
.os-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}

/* TECH BANNER */
.tech-banner{display:flex;align-items:center;gap:8px;padding:7px 40px;background:rgba(8,7,20,.6);border-bottom:1px solid var(--rim);overflow-x:auto;flex-wrap:nowrap}
.tech-banner::-webkit-scrollbar{height:2px}
.tb-lbl{font-family:'DM Mono',monospace;font-size:8px;color:var(--text3);letter-spacing:.16em;text-transform:uppercase;white-space:nowrap;flex-shrink:0}
.tb-sep{width:1px;height:14px;background:var(--rim2);flex-shrink:0}
.tc{font-family:'DM Mono',monospace;font-size:8px;padding:3px 9px;border-radius:4px;white-space:nowrap;flex-shrink:0;border:1px solid;letter-spacing:.05em;cursor:default;transition:all .2s}
.tc:hover{filter:brightness(1.3);transform:translateY(-1px)}

.wrap{max-width:840px;margin:0 auto;padding:60px 28px 130px}

/* BUTTONS */
.btn{padding:13px 26px;border-radius:var(--r2);font-family:'Syne',sans-serif;font-weight:700;font-size:14px;cursor:pointer;border:none;display:inline-flex;align-items:center;gap:9px;transition:all .2s cubic-bezier(.16,1,.3,1);position:relative;overflow:hidden;letter-spacing:.01em}
.btn-prime{background:linear-gradient(135deg,#F97316 0%,#7C3AED 100%);color:#fff;box-shadow:0 4px 24px rgba(249,115,22,.35)}
.btn-prime:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 8px 40px rgba(249,115,22,.5)}
.btn-prime:disabled{opacity:.3;cursor:not-allowed}
.btn-em{background:linear-gradient(135deg,#059669,#0891B2);color:#fff;box-shadow:0 4px 20px rgba(5,150,105,.3)}
.btn-em:hover:not(:disabled){transform:translateY(-2px)}
.btn-ghost{background:var(--glass2);color:var(--text2);border:1px solid var(--rim2);backdrop-filter:blur(8px)}
.btn-ghost:hover{background:var(--glass3);color:var(--text)}
.btn-sm{padding:8px 16px;font-size:12px}

/* OLLAMA SETUP CARD */
.ollama-card{padding:20px 24px;border-radius:16px;border:1px solid rgba(249,115,22,.25);background:rgba(249,115,22,.04);margin-bottom:20px;animation:slide-up .4s ease both}
.oc-title{font-family:'DM Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.16em;color:var(--brand);margin-bottom:10px}
.oc-code{background:rgba(0,0,0,.5);border:1px solid rgba(249,115,22,.2);border-radius:10px;padding:10px 14px;font-family:'DM Mono',monospace;font-size:11px;color:var(--brand2);margin:5px 0;line-height:1.8}
.oc-code span{color:var(--text3);font-size:10px;display:block;margin-bottom:3px}
.oc-note{font-family:'DM Mono',monospace;font-size:10px;color:var(--text3);margin-top:8px;line-height:1.7}

/* HERO */
.hero{display:flex;flex-direction:column;align-items:center;text-align:center;padding:80px 0 60px}
.hero-eyebrow{display:inline-flex;align-items:center;gap:10px;font-family:'DM Mono',monospace;font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:var(--brand2);padding:8px 20px;border-radius:40px;margin-bottom:32px;background:rgba(249,115,22,.06);border:1px solid rgba(249,115,22,.2)}
.eyebrow-dot{width:6px;height:6px;border-radius:50%;background:var(--brand);animation:pulse-glow 2s ease infinite;box-shadow:0 0 8px var(--brand)}
.hero-h{font-size:clamp(48px,8vw,88px);font-weight:900;line-height:.9;letter-spacing:-4px;margin-bottom:20px}
.hero-h .l1{display:block;color:var(--text)}
.hero-h .l2{display:block;font-family:'Playfair Display',serif;font-style:italic;font-weight:700;font-size:clamp(54px,9vw,96px);background:linear-gradient(135deg,#F97316 0%,#A78BFA 40%,#F59E0B 80%,#FB7185 100%);background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:shimmer 4s linear 1s infinite;letter-spacing:-3px}
.hero-p{font-size:15px;line-height:1.8;color:var(--text2);max-width:480px;margin-bottom:48px}
.name-card{width:100%;max-width:400px;background:rgba(14,12,30,.7);border:1px solid var(--rim2);border-radius:22px;padding:36px;backdrop-filter:blur(40px);box-shadow:0 40px 100px rgba(0,0,0,.5),inset 0 1px 0 rgba(255,255,255,.05);position:relative;overflow:hidden}
.name-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(249,115,22,.5),rgba(124,58,237,.5),transparent)}
.nc-lbl{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.18em;text-transform:uppercase;color:var(--text3);margin-bottom:10px;display:block}
.nc-input{width:100%;padding:14px 18px;margin-bottom:18px;border:1px solid var(--rim2);border-radius:14px;background:rgba(0,0,0,.35);color:var(--text);font-family:'Syne',sans-serif;font-size:20px;font-weight:700;outline:none;transition:all .25s;text-align:center;letter-spacing:-.3px}
.nc-input:focus{border-color:rgba(249,115,22,.5);box-shadow:0 0 0 4px rgba(249,115,22,.1)}
.nc-input::placeholder{color:var(--text3);font-weight:400;font-size:16px}
.feat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:36px;max-width:640px}
.feat-item{display:flex;flex-direction:column;align-items:center;gap:6px;padding:14px 10px;border-radius:14px;border:1px solid var(--rim);background:var(--glass);transition:all .2s}
.feat-item:hover{border-color:rgba(249,115,22,.25);background:rgba(249,115,22,.05);transform:translateY(-2px)}
.feat-ico{font-size:20px}
.feat-txt{font-family:'DM Mono',monospace;font-size:8px;letter-spacing:.06em;text-transform:uppercase;color:var(--text3);text-align:center}

/* UPLOAD */
.up-header{margin-bottom:32px}
.up-greeting{font-family:'DM Mono',monospace;font-size:11px;color:var(--brand);letter-spacing:.08em;margin-bottom:6px;display:flex;align-items:center;gap:10px}
.up-greeting::before{content:'';display:block;width:20px;height:1px;background:var(--brand)}
.up-title{font-size:32px;font-weight:800;letter-spacing:-1.5px;margin-bottom:6px}
.up-sub{font-size:14px;color:var(--text2);line-height:1.7}

/* RAG PANEL */
.rag-panel{margin-bottom:16px;padding:14px 18px;border-radius:14px;border:1px solid rgba(249,115,22,.18);background:rgba(249,115,22,.03)}
.rag-title{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:.16em;color:var(--brand);margin-bottom:10px;display:flex;align-items:center;gap:8px}
.rdot{width:6px;height:6px;border-radius:50%;background:var(--brand);box-shadow:0 0 6px var(--brand);animation:pulse-glow 1.5s ease infinite}
.rag-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.rag-stat{background:rgba(0,0,0,.25);border-radius:8px;padding:8px;text-align:center;border:1px solid var(--rim)}
.rag-n{font-family:'DM Mono',monospace;font-size:18px;font-weight:500;color:var(--brand2)}
.rag-l{font-family:'DM Mono',monospace;font-size:8px;color:var(--text3);text-transform:uppercase;letter-spacing:.1em;margin-top:3px}

.dropzone{border:1px dashed rgba(249,115,22,.2);border-radius:18px;padding:50px 32px;text-align:center;cursor:pointer;transition:all .3s cubic-bezier(.16,1,.3,1);background:linear-gradient(135deg,rgba(249,115,22,.02),rgba(124,58,237,.01))}
.dropzone:hover,.dropzone.over{border-color:rgba(249,115,22,.45);border-style:solid;transform:translateY(-4px);box-shadow:0 16px 60px rgba(0,0,0,.3)}
.dz-icon{width:66px;height:66px;margin:0 auto 16px;border-radius:20px;background:linear-gradient(135deg,rgba(249,115,22,.15),rgba(124,58,237,.1));border:1px solid rgba(249,115,22,.2);display:flex;align-items:center;justify-content:center;font-size:28px;transition:all .3s}
.dropzone:hover .dz-icon{transform:scale(1.1) rotate(-5deg);border-color:rgba(249,115,22,.45)}
.dz-title{font-size:17px;font-weight:700;margin-bottom:4px}
.dz-sub{font-family:'DM Mono',monospace;font-size:11px;color:var(--text3);margin-bottom:12px}
.dz-chips{display:flex;gap:6px;justify-content:center;flex-wrap:wrap}
.dz-chip{font-family:'DM Mono',monospace;font-size:9px;padding:3px 10px;border-radius:20px;background:rgba(249,115,22,.07);border:1px solid var(--rim);color:var(--text3);letter-spacing:.05em}

.file-pill{display:flex;align-items:center;gap:12px;padding:12px 16px;border-radius:14px;border:1px solid rgba(249,115,22,.22);background:rgba(249,115,22,.05);margin-bottom:14px}
.fp-icon{width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,rgba(249,115,22,.3),rgba(124,58,237,.15));display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0}
.fp-info{flex:1}
.fp-name{font-weight:700;font-size:14px}
.fp-meta{font-family:'DM Mono',monospace;font-size:10px;color:var(--text3);margin-top:2px}
.fp-rm{width:28px;height:28px;border-radius:8px;border:1px solid var(--rim);background:transparent;color:var(--text3);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:12px;transition:all .2s}
.fp-rm:hover{border-color:#F87171;color:#F87171}

.doc-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--rim);border:1px solid var(--rim);border-radius:14px;overflow:hidden;margin-bottom:20px}
.ds{background:var(--ink2);padding:16px 12px;text-align:center}
.ds-n{font-size:22px;font-weight:800;color:var(--brand2);letter-spacing:-.5px}
.ds-l{font-family:'DM Mono',monospace;font-size:8px;color:var(--text3);text-transform:uppercase;letter-spacing:.1em;margin-top:4px}

.or-bar{display:flex;align-items:center;gap:12px;margin:18px 0;font-family:'DM Mono',monospace;font-size:10px;color:var(--text3);letter-spacing:.08em}
.or-bar::before,.or-bar::after{content:'';flex:1;height:1px;background:var(--rim)}
.paste-ta{width:100%;min-height:140px;padding:16px 18px;border:1px solid var(--rim);border-radius:14px;background:var(--ink1);color:var(--text);resize:vertical;font-family:'Syne',sans-serif;font-size:14px;line-height:1.75;outline:none;transition:all .25s}
.paste-ta:focus{border-color:rgba(249,115,22,.4);box-shadow:0 0 0 3px rgba(249,115,22,.08)}
.paste-ta::placeholder{color:var(--text3)}

.topic-card{margin-top:20px;padding:16px 20px;border-radius:14px;border:1px solid rgba(249,115,22,.18);background:rgba(249,115,22,.03)}
.tc-hd{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:.16em;color:var(--brand);margin-bottom:12px;display:flex;align-items:center;gap:10px}
.tc-hd::after{content:'';flex:1;height:1px;background:rgba(249,115,22,.2)}
.tc-tags{display:flex;flex-wrap:wrap;gap:7px}
.tc-tag{font-size:12px;font-weight:600;padding:5px 12px;border-radius:30px;background:rgba(249,115,22,.07);border:1px solid rgba(249,115,22,.18);color:var(--brand2)}

.cfg-sec{margin-top:28px}
.cfg-title{font-size:17px;font-weight:800;letter-spacing:-.4px;margin-bottom:16px;display:flex;align-items:center;gap:10px}
.cfg-title::after{content:'';flex:1;height:1px;background:var(--rim)}
.cfg{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.cfg-block{display:flex;flex-direction:column;gap:7px}
.cfg-lbl{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:.14em;color:var(--text3)}
.cfg-sel{width:100%;padding:11px 14px;border:1px solid var(--rim);border-radius:12px;background:var(--ink2);color:var(--text);font-family:'Syne',sans-serif;font-size:13px;font-weight:600;outline:none;cursor:pointer;transition:all .2s;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%233D3860' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center}
.cfg-sel:focus{border-color:rgba(249,115,22,.4)}
.sec-div{height:1px;background:var(--rim);margin:24px 0}

/* LOADING */
.load-screen{text-align:center;padding:90px 24px}
.loader-ring{width:68px;height:68px;margin:0 auto 28px;position:relative}
.lr-o{position:absolute;inset:0;border:2px solid transparent;border-top-color:var(--brand);border-right-color:rgba(249,115,22,.4);border-radius:50%;animation:spin .9s linear infinite;box-shadow:0 0 28px rgba(249,115,22,.25)}
.lr-i{position:absolute;inset:10px;border:1.5px solid transparent;border-bottom-color:var(--violet2);border-left-color:rgba(124,58,237,.35);border-radius:50%;animation:spin-rev .6s linear infinite}
.lr-d{position:absolute;top:50%;left:50%;width:8px;height:8px;border-radius:50%;background:var(--brand2);transform:translate(-50%,-50%);box-shadow:0 0 14px var(--brand);animation:pulse-glow 1s ease infinite}
.load-title{font-size:28px;font-weight:800;letter-spacing:-1px;margin-bottom:6px}
.load-sub{font-size:14px;color:var(--text2);margin-bottom:28px}
.ls-item{display:flex;align-items:center;gap:10px;font-family:'DM Mono',monospace;font-size:11px;color:var(--text3);margin-bottom:8px;animation:slide-up .4s ease both}
.ls-dot{width:5px;height:5px;border-radius:50%;background:var(--brand);box-shadow:0 0 8px var(--brand);animation:pulse-glow 1.2s ease infinite}

/* QUIZ */
.quiz-header{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:32px;gap:16px;flex-wrap:wrap}
.qh-greeting{font-family:'DM Mono',monospace;font-size:10px;color:var(--brand);letter-spacing:.08em;margin-bottom:5px}
.qh-title{font-size:30px;font-weight:800;letter-spacing:-1px;margin-bottom:3px}
.qh-meta{font-size:13px;color:var(--text3)}
.qh-right{display:flex;flex-direction:column;align-items:flex-end;gap:7px}
.timer-w{font-family:'DM Mono',monospace;font-size:22px;font-weight:500;color:var(--gold);padding:9px 18px;border-radius:12px;background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.2);letter-spacing:.04em}
.timer-w.urgent{color:#F87171;background:rgba(248,113,113,.07);border-color:rgba(248,113,113,.25);animation:pulse-glow 1s ease infinite}
.prog-chip{font-family:'DM Mono',monospace;font-size:10px;padding:5px 14px;border-radius:30px;color:var(--text2);background:rgba(249,115,22,.08);border:1px solid rgba(249,115,22,.2)}
.prog-chip strong{color:var(--brand2)}

.q-card{border-radius:18px;border:1px solid var(--rim);background:var(--ink1);overflow:hidden;margin-bottom:14px;transition:all .25s;animation:slide-up .5s cubic-bezier(.16,1,.3,1) both;position:relative}
.q-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(249,115,22,.3),transparent);opacity:0;transition:opacity .3s}
.q-card:hover::before,.q-card.answered::before{opacity:1}
.q-card:hover{border-color:var(--rim2)}
.q-card.answered{border-color:rgba(249,115,22,.2)}
.q-card.flagged{border-color:rgba(245,158,11,.3)}
.q-head{display:flex;align-items:center;gap:9px;padding:14px 18px 12px;border-bottom:1px solid var(--rim);background:rgba(0,0,0,.2)}
.q-num{width:32px;height:32px;border-radius:9px;flex-shrink:0;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,rgba(249,115,22,.25),rgba(124,58,237,.1));border:1px solid rgba(249,115,22,.2);font-size:12px;font-weight:800;color:var(--brand2)}
.q-badge{font-family:'DM Mono',monospace;font-size:9px;padding:3px 10px;border-radius:20px;letter-spacing:.07em;text-transform:uppercase}
.q-badge.topic{background:rgba(249,115,22,.07);border:1px solid rgba(249,115,22,.15);color:var(--brand2)}
.q-badge.type{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.15);color:var(--gold)}
.q-check{margin-left:auto;font-family:'DM Mono',monospace;font-size:9px;color:var(--emerald);display:flex;align-items:center;gap:5px}
.q-check::before{content:'';width:5px;height:5px;border-radius:50%;background:var(--emerald)}
.q-flag{width:28px;height:28px;border-radius:7px;border:1px solid var(--rim);background:transparent;color:var(--text3);cursor:pointer;font-size:13px;display:flex;align-items:center;justify-content:center;transition:all .2s}
.q-flag:hover{border-color:rgba(245,158,11,.4);color:var(--gold)}
.q-flag.on{border-color:rgba(245,158,11,.4);background:rgba(245,158,11,.08);color:var(--gold)}
.q-body{padding:20px 18px}
.q-text{font-size:16px;font-weight:700;line-height:1.55;margin-bottom:18px}

.opts{display:flex;flex-direction:column;gap:7px}
.opt{display:flex;align-items:flex-start;gap:11px;padding:12px 14px;border-radius:11px;border:1px solid var(--rim);background:var(--ink2);cursor:pointer;font-size:14px;line-height:1.5;font-weight:500;transition:all .15s;color:var(--text);width:100%;font-family:'Syne',sans-serif;text-align:left}
.opt:hover:not(:disabled){border-color:rgba(249,115,22,.35);background:rgba(249,115,22,.06);transform:translateX(4px)}
.opt.sel{border-color:rgba(249,115,22,.45);background:rgba(249,115,22,.08)}
.opt.corr{border-color:rgba(16,185,129,.45);background:rgba(16,185,129,.07)}
.opt.wrong{border-color:rgba(248,113,113,.35);background:rgba(248,113,113,.06)}
.opt:disabled{cursor:default}
.opt-key{width:24px;height:24px;border-radius:7px;flex-shrink:0;border:1px solid var(--rim);background:var(--ink3);font-family:'DM Mono',monospace;font-size:10px;display:flex;align-items:center;justify-content:center;transition:all .15s;margin-top:1px}
.opt.sel .opt-key{border-color:var(--brand);background:var(--brand);color:#fff}
.opt.corr .opt-key{border-color:var(--emerald);background:var(--emerald);color:#fff}
.opt.wrong .opt-key{border-color:#F87171;background:#F87171;color:#fff}
.short-ta{width:100%;min-height:110px;padding:14px 16px;border:1px solid var(--rim);border-radius:12px;background:var(--ink2);color:var(--text);resize:vertical;font-family:'Syne',sans-serif;font-size:14px;line-height:1.7;outline:none;transition:all .2s}
.short-ta:focus{border-color:rgba(249,115,22,.4);box-shadow:0 0 0 3px rgba(249,115,22,.08)}
.short-ta:disabled{opacity:.7}
.hint-row{display:flex;align-items:flex-start;gap:7px;margin-top:8px;padding:9px 12px;border-radius:9px;background:rgba(245,158,11,.05);border:1px solid rgba(245,158,11,.12)}
.hint-txt{font-family:'DM Mono',monospace;font-size:11px;color:rgba(245,158,11,.8);line-height:1.6}
.conf-row{display:flex;align-items:center;gap:7px;margin-top:14px;flex-wrap:wrap}
.conf-label{font-family:'DM Mono',monospace;font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.12em}
.conf-btn{padding:4px 12px;border-radius:20px;border:1px solid var(--rim);background:transparent;font-family:'Syne',sans-serif;font-size:11px;font-weight:700;cursor:pointer;transition:all .18s;color:var(--text3)}
.conf-btn.high{background:rgba(16,185,129,.1);border-color:rgba(16,185,129,.2);color:var(--emerald)}
.conf-btn.high.on{background:var(--emerald);color:#fff}
.conf-btn.med{background:rgba(245,158,11,.1);border-color:rgba(245,158,11,.2);color:var(--gold)}
.conf-btn.med.on{background:var(--gold);color:#fff}
.conf-btn.low{background:rgba(248,113,113,.1);border-color:rgba(248,113,113,.2);color:#F87171}
.conf-btn.low.on{background:#F87171;color:#fff}

/* SUBMIT BAR */
.sub-bar{position:fixed;bottom:0;left:0;right:0;z-index:100;background:rgba(3,2,10,.9);backdrop-filter:blur(32px);border-top:1px solid var(--rim);padding:13px 40px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.sb-prog{flex:1;min-width:120px}
.sb-bar{height:3px;background:var(--ink3);border-radius:2px;overflow:hidden;margin-bottom:6px}
.sb-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--brand),var(--violet));transition:width .4s cubic-bezier(.16,1,.3,1)}
.sb-meta{font-family:'DM Mono',monospace;font-size:10px;color:var(--text3)}
.sb-meta strong{color:var(--brand2)}
.sb-flags{font-family:'DM Mono',monospace;font-size:10px;color:var(--gold)}
.sb-actions{display:flex;gap:8px}

/* EVAL OVERLAY */
.eval-overlay{position:fixed;inset:0;z-index:300;background:rgba(3,2,10,.96);backdrop-filter:blur(24px);display:flex;flex-direction:column;align-items:center;justify-content:center;animation:fade-in .4s ease both}
.eval-ring{width:76px;height:76px;margin:0 auto 28px;position:relative}
.er-o{position:absolute;inset:0;border:3px solid transparent;border-top-color:var(--brand);border-right-color:rgba(249,115,22,.4);border-radius:50%;animation:spin .9s linear infinite;box-shadow:0 0 36px rgba(249,115,22,.3)}
.er-m{position:absolute;inset:12px;border:2px solid transparent;border-bottom-color:var(--violet2);border-left-color:rgba(124,58,237,.4);border-radius:50%;animation:spin-rev .6s linear infinite}
.er-i{position:absolute;inset:24px;border:1px solid transparent;border-top-color:var(--rose);border-radius:50%;animation:spin 1.5s linear infinite}
.eval-title{font-size:26px;font-weight:800;letter-spacing:-.8px;margin-bottom:6px}
.eval-sub{font-size:13px;color:var(--text2);margin-bottom:28px}
.eval-prog{width:360px;max-width:90vw}
.eval-bar-wrap{height:5px;background:var(--ink3);border-radius:3px;overflow:hidden;margin-bottom:10px}
.eval-bar{height:100%;background:linear-gradient(90deg,var(--brand),var(--violet));border-radius:3px;transition:width .6s cubic-bezier(.16,1,.3,1)}
.eval-step{font-family:'DM Mono',monospace;font-size:11px;color:var(--text3);text-align:center}

/* RESULTS */
.results-header{text-align:center;padding:60px 0 48px}
.rank-shell{display:flex;flex-direction:column;align-items:center;justify-content:center;width:148px;height:148px;border-radius:50%;margin:0 auto 24px;position:relative;border:2px solid currentColor;animation:pop .7s cubic-bezier(.34,1.56,.64,1) .2s both}
.rank-shell::before{content:'';position:absolute;inset:-6px;border-radius:50%;background:conic-gradient(from 0deg,transparent 0%,currentColor 20%,transparent 40%);opacity:.2;animation:halo 6s linear infinite}
.rank-glow{position:absolute;inset:-30px;border-radius:50%;background:radial-gradient(circle,var(--glow,transparent) 0%,transparent 70%);opacity:.5}
.rank-letter{font-size:52px;font-weight:900;line-height:1;letter-spacing:-3px;position:relative;z-index:1}
.rank-pct{font-family:'DM Mono',monospace;font-size:11px;opacity:.65;margin-top:2px;position:relative;z-index:1}
.res-emoji{font-size:32px;margin-bottom:10px;display:block;animation:float 3s ease infinite}
.res-name{font-family:'DM Mono',monospace;font-size:11px;color:var(--text3);margin-bottom:6px}
.res-rank-label{font-size:38px;font-weight:900;letter-spacing:-2px;margin-bottom:8px}
.res-pts{font-size:14px;color:var(--text2)}

.stat-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--rim);border:1px solid var(--rim);border-radius:16px;overflow:hidden;margin:28px 0}
.stat{background:var(--ink1);padding:20px 10px;text-align:center}
.stat-n{font-size:28px;font-weight:800;letter-spacing:-1px;line-height:1}
.stat-n.g{color:var(--emerald)}.stat-n.y{color:var(--gold)}.stat-n.r{color:#F87171}.stat-n.p{color:var(--brand2)}.stat-n.b{color:var(--sky)}
.stat-l{font-family:'DM Mono',monospace;font-size:8px;color:var(--text3);text-transform:uppercase;letter-spacing:.1em;margin-top:5px}

.section-hd{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:.16em;color:var(--text3);margin:28px 0 16px;display:flex;align-items:center;gap:12px}
.section-hd::after{content:'';flex:1;height:1px;background:var(--rim)}

.tb-row{display:flex;align-items:center;gap:11px;margin-bottom:9px}
.tb-name{font-size:13px;font-weight:600;min-width:130px;color:var(--text2)}
.tb-wrap{flex:1;height:7px;background:var(--ink3);border-radius:4px;overflow:hidden}
.tb-fill{height:100%;border-radius:4px;animation:bar-fill 1s cubic-bezier(.16,1,.3,1)}
.tb-pct{font-family:'DM Mono',monospace;font-size:10px;color:var(--text3);min-width:34px;text-align:right}

.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.chart-card{background:var(--ink1);border:1px solid var(--rim);border-radius:16px;padding:20px}
.chart-card.full{grid-column:1/-1}
.chart-title{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:.14em;color:var(--text3);margin-bottom:14px;display:flex;align-items:center;gap:7px}
.chart-title::before{content:'';width:3px;height:11px;border-radius:2px;background:var(--brand)}

.ai-card{padding:26px 30px;border-radius:18px;border:1px solid rgba(249,115,22,.18);background:linear-gradient(135deg,rgba(249,115,22,.04),rgba(124,58,237,.02));margin-bottom:28px;position:relative;overflow:hidden}
.ai-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(249,115,22,.4),rgba(124,58,237,.4),transparent)}
.ai-lbl{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:.16em;color:var(--brand2);margin-bottom:12px;display:flex;align-items:center;gap:10px}
.ai-lbl::before{content:'';width:20px;height:1px;background:var(--brand2)}
.ai-txt{font-size:14px;line-height:1.85;color:var(--text2)}

.reco-grid{display:grid;grid-template-columns:1fr 1fr;gap:11px;margin-bottom:28px}
.reco-card{padding:16px 18px;border-radius:14px;border:1px solid var(--rim);background:var(--ink2);transition:all .2s}
.reco-card:hover{border-color:var(--rim2);transform:translateY(-2px)}
.reco-type{font-family:'DM Mono',monospace;font-size:8px;letter-spacing:.12em;text-transform:uppercase;margin-bottom:5px}
.reco-topic{font-size:13px;font-weight:700;margin-bottom:3px}
.reco-action{font-size:12px;color:var(--text3);line-height:1.6}
.reco-prio{font-family:'DM Mono',monospace;font-size:8px;padding:2px 7px;border-radius:4px;margin-top:8px;display:inline-block}

.export-row{display:flex;align-items:center;gap:10px;margin-bottom:24px;padding:12px 16px;border-radius:12px;background:var(--glass);border:1px solid var(--rim);flex-wrap:wrap}
.er-lbl{font-family:'DM Mono',monospace;font-size:10px;color:var(--text3);flex:1}

.rev-card{margin-bottom:11px;padding:20px;border-radius:16px;border:1px solid var(--rim);background:var(--ink1);animation:slide-up .4s ease both;position:relative;overflow:hidden}
.rev-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px}
.rev-card.correct::before{background:var(--emerald)}.rev-card.incorrect::before{background:#F87171}.rev-card.partial::before{background:var(--gold)}
.rev-card.correct{border-color:rgba(16,185,129,.18)}.rev-card.incorrect{border-color:rgba(248,113,113,.15)}.rev-card.partial{border-color:rgba(245,158,11,.18)}
.rev-pills{display:flex;align-items:center;gap:5px;flex-wrap:wrap;margin-bottom:10px}
.rpill{font-family:'DM Mono',monospace;font-size:9px;padding:3px 9px;border-radius:20px;text-transform:uppercase;letter-spacing:.07em}
.rpill.c{background:rgba(16,185,129,.1);color:var(--emerald)}.rpill.i{background:rgba(248,113,113,.1);color:#F87171}.rpill.p{background:rgba(245,158,11,.1);color:var(--gold)}.rpill.s{background:rgba(249,115,22,.1);color:var(--brand2)}
.rev-q{font-size:14px;font-weight:700;margin-bottom:7px;line-height:1.45}
.rev-ans{font-size:13px;color:var(--text3)}
.rev-ans em{color:var(--text2);font-style:italic}
.rev-fb{font-size:13px;color:var(--text3);line-height:1.7;margin-top:10px;padding-top:10px;border-top:1px solid var(--rim)}

.lb-card{padding:22px;border-radius:18px;border:1px solid var(--rim);background:var(--ink1);margin-top:28px}
.lb-hd{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:.16em;color:var(--text3);margin-bottom:14px}
.lb-row{display:flex;align-items:center;gap:11px;padding:10px 12px;border-radius:10px;margin-bottom:5px;background:var(--glass)}
.lb-row.me{background:rgba(249,115,22,.07);border:1px solid rgba(249,115,22,.18)}
.lb-pos{font-family:'DM Mono',monospace;font-size:12px;font-weight:700;width:24px;color:var(--text3)}
.lb-pos.g{color:var(--gold)}.lb-pos.s{color:#C0C0C0}.lb-pos.b{color:#CD7F32}
.lb-name{flex:1;font-weight:700;font-size:14px}
.lb-score{font-family:'DM Mono',monospace;font-size:11px;color:var(--brand2)}
.lb-badge{font-size:12px;font-weight:700;padding:2px 8px;border-radius:6px}
.res-acts{display:flex;gap:9px;margin-top:28px;flex-wrap:wrap}

.toast{position:fixed;bottom:88px;right:20px;z-index:400;padding:11px 18px;border-radius:12px;background:var(--ink2);border:1px solid var(--rim2);color:var(--text);font-family:'DM Mono',monospace;font-size:12px;box-shadow:0 20px 60px rgba(0,0,0,.6);animation:slide-up .3s cubic-bezier(.16,1,.3,1) both;display:flex;align-items:center;gap:9px;backdrop-filter:blur(20px)}

/* RAG NOTE — shown when tinyllama returns nothing parseable */
.parse-warn{padding:12px 16px;border-radius:12px;border:1px solid rgba(251,191,36,.3);background:rgba(251,191,36,.06);font-family:'DM Mono',monospace;font-size:11px;color:var(--gold);line-height:1.7;margin-bottom:14px}

@media(max-width:640px){
  .nav{padding:12px 14px}.nav-steps{display:none}
  .wrap{padding:32px 14px 130px}
  .stat-grid{grid-template-columns:repeat(3,1fr)}.cfg{grid-template-columns:1fr 1fr}
  .doc-stats{grid-template-columns:repeat(2,1fr)}.feat-grid{grid-template-columns:repeat(2,1fr)}
  .quiz-header{flex-direction:column}.sub-bar{padding:11px 14px}
  .rag-grid{grid-template-columns:repeat(2,1fr)}.chart-grid{grid-template-columns:1fr}
  .reco-grid{grid-template-columns:1fr}
}
`;

const TECH_CHIPS = [
  { l: "RAG",          c: "#818CF8", b: "rgba(129,140,248,.12)" },
  { l: "Chunking",     c: "#34D399", b: "rgba(52,211,153,.12)"  },
  { l: "TF-IDF",       c: "#E879F9", b: "rgba(232,121,249,.12)" },
  { l: "VectorDB",     c: "#38BDF8", b: "rgba(56,189,248,.12)"  },
  { l: "Ollama",       c: "#F97316", b: "rgba(249,115,22,.12)"  },
  { l: "tinyllama",    c: "#FED7AA", b: "rgba(254,215,170,.12)" },
  { l: "Local LLM",    c: "#FB923C", b: "rgba(251,146,60,.12)"  },
  { l: "100% Offline", c: "#4ADE80", b: "rgba(74,222,128,.12)"  },
  { l: "MongoDB",      c: "#4ADE80", b: "rgba(74,222,128,.12)"  },
  { l: "Recharts",     c: "#F472B6", b: "rgba(244,114,182,.12)" },
  { l: "Privacy-first",c: "#67E8F9", b: "rgba(103,232,249,.12)" },
];

const GEN_STEPS = [
  "Chunking document (RAG)…",
  "Building TF-IDF embeddings…",
  "Indexing vector store…",
  "Retrieving top-k chunks…",
  "Prompting tinyllama via Ollama…",
  "Parsing & validating questions…",
  "Finalising quiz…",
];

// ─────────────────────────────────────────────────────────────────────────
//  MAIN APP
// ─────────────────────────────────────────────────────────────────────────
export default function App() {
  const [stage, setStage]     = useState("name");
  const [userName, setUser]   = useState("");
  const [docText, setDocText] = useState("");
  const [paste, setPaste]     = useState("");
  const [fileName, setFile]   = useState("");
  const [numQ, setNumQ]       = useState("5");
  const [qType, setQType]     = useState("mcq");       // default mcq — simpler for tinyllama
  const [difficulty, setDiff] = useState("easy");      // default easy
  const [timerOn, setTimerOn] = useState(false);
  const [timeLimit, setTimeL] = useState("300");
  const [drag, setDrag]       = useState(false);
  const [genStep, setGenStep] = useState(0);
  const [evalPct, setEvalPct] = useState(0);
  const [evalTxt, setEvalTxt] = useState("");
  const [questions, setQ]     = useState([]);
  const [topics, setTopics]   = useState([]);
  const [answers, setAns]     = useState({});
  const [confidence, setConf] = useState({});
  const [flagged, setFlagged] = useState({});
  const [feedbacks, setFb]    = useState({});
  const [summary, setSummary] = useState("");
  const [timer, setTimer]     = useState(0);
  const [timerActive, setTA]  = useState(false);
  const [toast, setToast]     = useState(null);
  const [lb, setLB]           = useState(LEADERBOARD);
  const [ragInfo, setRagInfo] = useState(null);
  const [recos, setRecos]     = useState([]);
  const [ollamaStatus, setOS] = useState("checking");
  const [ollamaModels, setOM] = useState([]);
  const [model, setModel]     = useState(OLLAMA_MODEL);
  const [parseWarn, setPW]    = useState("");

  const fileRef  = useRef();
  const timerRef = useRef();

  const msg = (m, icon = "ℹ") => { setToast({ m, icon }); setTimeout(() => setToast(null), 3800); };

  // ── Ollama check ──
  useEffect(() => {
    (async () => {
      const { ok, models } = await checkOllama();
      setOS(ok ? "connected" : "disconnected");
      if (ok && models.length) {
        setOM(models);
        const m = models.find(x => x.includes("mistral")) || models.find(x => x.includes("tinyllama")) || models[0];
        setModel(m);
      }
    })();
  }, []);

  // ── Timer ──
  useEffect(() => {
    if (timerActive && timer > 0) {
      timerRef.current = setTimeout(() => setTimer(t => t - 1), 1000);
    } else if (timerActive && timer === 0) {
      msg("⏱ Time's up! Submitting…", "⏰");
      evaluateAll();
    }
    return () => clearTimeout(timerRef.current);
  }, [timer, timerActive]);

  // ── File ──
  const handleFile = async f => {
    if (!f) return;
    const ok = f.type.startsWith("text/") || /\.(txt|md|csv|json|log|js|py|html|xml|yaml|yml)$/i.test(f.name);
    if (!ok) { msg("Text files only (.txt, .md, etc.)", "⚠"); return; }
    try {
      const t = await readFileAsText(f);
      setDocText(t); setFile(f.name);
    } catch { msg("Could not read file", "❌"); }
  };

  const activeText = docText || paste;
  const wordCount  = activeText.split(/\s+/).filter(Boolean).length;
  const sentCount  = activeText.split(/[.!?]+/).filter(Boolean).length;
  const paraCount  = activeText.split(/\n\n+/).filter(Boolean).length;
  const charCount  = activeText.length;

  // ── Auto-index whenever text changes ──
  useEffect(() => {
    if (activeText.length < 100) return;
    const t = setTimeout(() => {
      const chunks = chunkText(activeText);
      const vocab  = buildVocab(chunks);
      const idf    = buildIdf(chunks, vocab);
      vs.insertMany(chunks, vocab, idf);
      setRagInfo({ chunks: chunks.length, vocab: Object.keys(vocab).length });
      console.log("[MongoDB]", `quizai_rag.chunks — ${chunks.length} documents`);
    }, 800);
    return () => clearTimeout(t);
  }, [activeText]);

  const extractTopics = async () => {
    const text = activeText.trim().slice(0, 1500);
    if (text.length < 80) return;
    try {
      const raw = await ollama(
        `List 5 key topics from this text as a JSON array of short strings. Output ONLY the array.\n\n${text}`,
        "Output only a valid JSON array of strings, nothing else.", 200
      );
      const arr = parseJSON(raw);
      if (Array.isArray(arr)) setTopics(arr.slice(0, 8));
    } catch {}
  };

  // ── GENERATE ──
  const generate = async () => {
    if (ollamaStatus !== "connected") { msg("Ollama is not running. Start with: OLLAMA_ORIGINS=* ollama serve", "❌"); return; }
    const text = activeText.trim();
    if (!text || text.length < 50) { msg("Need at least 50 characters of text", "⚠"); return; }
    setStage("gen-loading"); setGenStep(0); setPW("");
    const iv = setInterval(() => setGenStep(s => Math.min(s + 1, GEN_STEPS.length - 1)), 800);

    // RAG — retrieve relevant chunks
    const query     = topics.slice(0, 3).join(" ") || text.slice(0, 150);
    const retrieved = vs.search(query, 6);
    const context   = retrieved.length > 0
      ? retrieved.map(d => d.text).join("\n\n---\n\n")
      : text.slice(0, 3000);

    const prompt = buildQuizPrompt(context, numQ, qType, difficulty);

    try {
      const raw = await ollama(prompt, "", 2048);
      clearInterval(iv);

      let qs = parseJSON(raw);

      // ── Validate & sanitise every question ──
      if (Array.isArray(qs) && qs.length > 0) {
        qs = qs.map((q, i) => {
          // ensure required fields
          const type = (q.type === "short") ? "short" : "mcq";
          const diff = normDiff(q.difficulty);
          const topic = (q.topic && typeof q.topic === "string") ? q.topic : "General";
          const question = (q.question && typeof q.question === "string") ? q.question : `Question ${i + 1}`;

          if (type === "mcq") {
            const options = Array.isArray(q.options) && q.options.length >= 2
              ? q.options.map(String)
              : ["True", "False", "Not mentioned", "Cannot determine"];
            // ensure answer is one of the options
            const answer = options.includes(q.answer) ? q.answer : options[0];
            return { id: i + 1, type, topic, difficulty: diff, question, options, answer, explanation: q.explanation || "" };
          } else {
            return {
              id: i + 1, type: "short", topic, difficulty: diff, question,
              answer: q.answer || "",
              keywords: Array.isArray(q.keywords) ? q.keywords.map(String) : [],
              hint: q.hint || "",
            };
          }
        });
      }

      if (!Array.isArray(qs) || qs.length === 0) {
        // tinyllama fallback: build basic MCQ from first sentence pairs
        setPW("tinyllama returned unparseable output — auto-generating fallback MCQ questions from document.");
        qs = buildFallbackQuestions(activeText, parseInt(numQ));
      }

      setQ(qs); setAns({}); setFb({}); setConf({}); setFlagged({}); setSummary(""); setRecos([]);
      if (timerOn) { setTimer(parseInt(timeLimit)); setTA(true); }
      setStage("quiz");
    } catch (e) {
      clearInterval(iv);
      msg("Ollama error: " + e.message, "❌");
      setStage("upload");
    }
  };

  // ── Fallback question builder (no LLM) ─ used when tinyllama fails ──
  const buildFallbackQuestions = (text, n) => {
    const sentences = text.split(/[.!?]+/).map(s => s.trim()).filter(s => s.split(/\s+/).length > 6).slice(0, 20);
    const qs = [];
    for (let i = 0; i < Math.min(n, sentences.length); i++) {
      const s = sentences[i];
      const words = s.split(/\s+/);
      // blank one important word
      const targetIdx = Math.floor(words.length * 0.5);
      const answer = words[targetIdx];
      const blanked = [...words.slice(0, targetIdx), "___", ...words.slice(targetIdx + 1)].join(" ");
      const distractors = words
        .filter((_, j) => j !== targetIdx && words[j].length > 3)
        .sort(() => Math.random() - 0.5)
        .slice(0, 3);
      while (distractors.length < 3) distractors.push("None of the above");
      const options = [answer, ...distractors].sort(() => Math.random() - 0.5);
      qs.push({
        id: i + 1, type: "mcq", topic: "Document", difficulty: "easy",
        question: `Fill in the blank: "${blanked}"`,
        options, answer, explanation: `The original text states: "${s}"`,
      });
    }
    return qs.length ? qs : [{
      id: 1, type: "mcq", topic: "Document", difficulty: "easy",
      question: "Did you read the document provided?",
      options: ["Yes, I read it", "No, I did not", "Partially", "I skimmed it"],
      answer: "Yes, I read it", explanation: "This is a default fallback question.",
    }];
  };

  const answeredCount = Object.keys(answers).filter(k => answers[k] !== undefined && answers[k] !== "").length;
  const flaggedCount  = Object.values(flagged).filter(Boolean).length;

  // ── EVALUATE ──
  const evaluateAll = async () => {
    clearTimeout(timerRef.current); setTA(false);
    setStage("eval-loading"); setEvalPct(0);
    const fb = {};
    const mcqQs   = questions.filter(q => q.type === "mcq");
    const shortQs = questions.filter(q => q.type === "short");
    let done = 0;
    const total = 4 + shortQs.length;
    const tick  = txt => { done++; setEvalPct(Math.round((done / total) * 100)); setEvalTxt(txt); };

    tick("Parsing MCQ responses…");
    await new Promise(r => setTimeout(r, 250));
    mcqQs.forEach(q => {
      const idx = questions.indexOf(q);
      const ua  = answers[idx] || "";
      const ok  = ua.trim() === (q.answer || "").trim();
      fb[idx] = {
        status: ok ? "correct" : "incorrect",
        text: ok ? (q.explanation || "Correct!") : `Correct: ${q.answer}. ${q.explanation || ""}`.trim(),
        score: ok ? 10 : 0,
        modelAns: q.answer,
      };
    });

    tick("Grading short answers…");
    for (const q of shortQs) {
      const idx = questions.indexOf(q);
      const ua  = (answers[idx] || "").trim();
      if (!ua) { fb[idx] = { status: "incorrect", text: "No answer provided.", score: 0, modelAns: q.answer }; continue; }

      // keyword-based scoring (reliable, no LLM needed)
      const kws  = (q.keywords || []).map(k => k.toLowerCase());
      const hits = kws.filter(k => ua.toLowerCase().includes(k)).length;
      const base = kws.length > 0 ? hits / kws.length : 0.5;

      // length heuristic
      const len  = ua.split(/\s+/).length;
      const lenBonus = len >= 10 ? 0.1 : 0;
      const score = Math.min(10, Math.round((base + lenBonus) * 10));
      const status = score >= 7 ? "correct" : score >= 4 ? "partial" : "incorrect";

      // optionally ask tinyllama to grade — with generous timeout
      let aiText = "";
      try {
        const raw = await ollama(
          `Question: ${q.question}\nExpected: ${q.answer}\nStudent: ${ua}\n\nIn ONE sentence, say if this answer is correct, partial, or incorrect, and why.`,
          "You are a strict examiner. Reply in ONE sentence only.", 180
        );
        aiText = raw.trim().slice(0, 300);
      } catch {}

      fb[idx] = { status, text: aiText || `Keyword match: ${hits}/${kws.length || 1} terms. Score: ${score}/10`, score, modelAns: q.answer };
    }
    setFb(fb);

    tick("Computing topic breakdown…");
    await new Promise(r => setTimeout(r, 200));
    const totPts = Object.values(fb).reduce((s, f) => s + (f?.score || 0), 0);
    const maxPts = questions.length * 10;
    const pct    = maxPts > 0 ? Math.round((totPts / maxPts) * 100) : 0;

    // ── AI summary ──
    tick("Generating performance summary…");
    try {
      const info = questions.map((q, i) => `Q${i + 1}(${q.topic}):${fb[i]?.status || "?"}`).join(", ");
      const t = await ollama(
        `Write 2-3 sentences of feedback for ${userName} who scored ${pct}%.\nResults: ${info}.\nAddress them by name. Be encouraging and specific.`,
        "You are a supportive tutor. Write only the feedback, no headers.", 300
      );
      setSummary(t.trim());
    } catch { setSummary(`${userName}, you scored ${pct}%. Review the questions below to strengthen your understanding.`); }

    // ── AI recommendations ──
    try {
      const tbMap = {};
      questions.forEach((q, i) => {
        if (!tbMap[q.topic]) tbMap[q.topic] = { total: 0, got: 0 };
        tbMap[q.topic].total += 10;
        tbMap[q.topic].got   += fb[i]?.score || 0;
      });
      const weak   = Object.entries(tbMap).filter(([, v]) => v.total > 0 && (v.got / v.total) < 0.6).map(([t]) => t).join(", ");
      const strong = Object.entries(tbMap).filter(([, v]) => v.total > 0 && (v.got / v.total) >= 0.8).map(([t]) => t).join(", ");
      const raw = await ollama(
        `Student: ${userName}, score: ${pct}%. Weak: ${weak || "none"}. Strong: ${strong || "none"}.\nReturn a JSON array of 4 recommendations:\n[{"type":"review","topic":"x","action":"1 sentence","priority":"high"},...]`,
        "Return only valid JSON array, nothing else.", 600
      );
      const arr = parseJSON(raw);
      if (Array.isArray(arr)) setRecos(arr.slice(0, 4));
    } catch {}

    // MongoDB save
    MongoDB.save({
      _id: `s_${Date.now()}`, user: userName, score: pct, pts: totPts, max: maxPts,
      rank: getRank(pct).rank, model, ragChunks: ragInfo?.chunks || 0,
      timestamp: new Date().toISOString(),
    });

    const rank = getRank(pct);
    saveLB({ name: userName, pct, pts: totPts, max: maxPts, rank: rank.rank, color: rank.color });
    setLB([...LEADERBOARD]);
    setStage("results");
  };

  // ── Derived stats ──
  // SAFE topicBreakdown — guards against undefined entries
  const topicBreakdown = () => {
    const map = {};
    questions.forEach((q, i) => {
      const topic = (q && q.topic) ? q.topic : "General";
      if (!map[topic]) map[topic] = { total: 0, got: 0 };
      map[topic].total += 10;
      map[topic].got   += (feedbacks[i]?.score ?? 0);
    });
    return Object.entries(map).map(([t, { total, got }]) => ({
      topic: t,
      pct: total > 0 ? Math.round((got / total) * 100) : 0,
      got, total,
    }));
  };

  const totalPts = Object.values(feedbacks).reduce((s, f) => s + (f?.score || 0), 0);
  const maxPts   = questions.length * 10;
  const pct      = maxPts > 0 ? Math.round((totalPts / maxPts) * 100) : 0;
  const nCorr    = Object.values(feedbacks).filter(f => f?.status === "correct").length;
  const nPart    = Object.values(feedbacks).filter(f => f?.status === "partial").length;
  const nWrong   = Object.values(feedbacks).filter(f => f?.status === "incorrect").length;
  const rank     = getRank(pct);

  // SAFE chart data — useMemo with guard on feedbacks & questions
  const radarData = useMemo(() => {
    if (!questions.length) return [];
    return topicBreakdown().map(t => ({ topic: t.topic.slice(0, 12), score: t.pct, full: 100 }));
  }, [feedbacks, questions]);

  const barData = useMemo(() => [
    { name: "Correct", value: nCorr, fill: "#34D399" },
    { name: "Partial",  value: nPart, fill: "#FBBF24" },
    { name: "Wrong",    value: nWrong, fill: "#F87171" },
  ], [nCorr, nPart, nWrong]);

  // SAFE diffData — normalises difficulty before grouping
  const diffData = useMemo(() => {
    const map = { easy: { correct: 0, total: 0 }, medium: { correct: 0, total: 0 }, hard: { correct: 0, total: 0 } };
    questions.forEach((q, i) => {
      const d = normDiff(q?.difficulty);   // ← FIX: always a valid key
      map[d].total++;
      if (feedbacks[i]?.status === "correct") map[d].correct++;
    });
    return Object.entries(map).map(([d, v]) => ({
      diff: d,
      pct: v.total > 0 ? Math.round((v.correct / v.total) * 100) : 0,
      total: v.total,
    }));
  }, [feedbacks, questions]);

  const exportResults = () => {
    const tb = topicBreakdown();
    const lines = [
      `QUIZ RESULTS — ${userName}`, `Date: ${new Date().toLocaleString()}`,
      `Model: ${model} (Ollama — Local)`,
      `Score: ${totalPts}/${maxPts} (${pct}%) — Rank: ${rank.rank} (${rank.label})`,
      `Correct: ${nCorr} | Partial: ${nPart} | Incorrect: ${nWrong}`,
      `RAG: ${ragInfo?.chunks || 0} chunks, ${ragInfo?.vocab || 0} vocab terms`,
      `MongoDB sessions: ${MongoDB.sessions.length}`, "",
      "=== TOPIC BREAKDOWN ===",
      ...tb.map(t => `${t.topic}: ${t.pct}% (${t.got}/${t.total})`), "",
      "=== RECOMMENDATIONS ===",
      ...recos.map(r => `[${r.type?.toUpperCase()}] ${r.topic}: ${r.action}`), "",
      "=== AI SUMMARY ===", summary, "",
      "=== QUESTION REVIEW ===",
      ...questions.map((q, i) => {
        const fb = feedbacks[i];
        return [
          `Q${i + 1} [${q.topic}/${q.difficulty}] — ${fb?.status || "?"} (${fb?.score || 0}/10)`,
          `Q: ${q.question}`,
          `Your: ${answers[i] || "(none)"}`,
          q.type === "mcq" ? `Correct: ${q.answer}` : "",
          `Feedback: ${fb?.text || ""}`, "",
        ].filter(Boolean).join("\n");
      }),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `quiz_${userName}_${Date.now()}.txt`;
    a.click();
    msg("Exported!", "✅");
  };

  const restart = () => {
    setStage("name"); setUser(""); setDocText(""); setPaste(""); setFile("");
    setQ([]); setAns({}); setFb({}); setConf({}); setFlagged({}); setSummary("");
    setTopics([]); setTA(false); setTimer(0); setRagInfo(null); setRecos([]); setPW("");
  };
  const retake = () => {
    setAns({}); setFb({}); setConf({}); setFlagged({}); setSummary(""); setRecos([]);
    if (timerOn) { setTimer(parseInt(timeLimit)); setTA(true); }
    setStage("quiz");
  };

  const si          = { name: 0, upload: 1, "gen-loading": 2, quiz: 2, "eval-loading": 3, results: 3 }[stage] ?? 0;
  const stageLabels = ["Profile", "Document", "Quiz", "Results"];

  const recoStyle = type => ({
    review:   { c: "#F87171", b: "rgba(248,113,113,.08)" },
    practice: { c: "#FBBF24", b: "rgba(251,191,36,.08)"  },
    advance:  { c: "#34D399", b: "rgba(52,211,153,.08)"  },
    resource: { c: "#818CF8", b: "rgba(129,140,248,.08)" },
  }[type] || { c: "#F87171", b: "rgba(248,113,113,.08)" });

  const prioStyle = p => ({
    high:   { c: "#F87171", b: "rgba(248,113,113,.12)" },
    medium: { c: "#FBBF24", b: "rgba(251,191,36,.1)"   },
    low:    { c: "#34D399", b: "rgba(52,211,153,.1)"   },
  }[p] || {});

  // ─── RENDER ───────────────────────────────────────────────────────────
  return (
    <>
      <style>{CSS}</style>

      <div className="bg-canvas">
        <div className="grid-bg"/>
        <div className="stars"/>
        <div className="nebula nb1"/><div className="nebula nb2"/><div className="nebula nb3"/>
        <div className="scanlines"/>
      </div>

      <div className="page">

        {/* Eval overlay */}
        {stage === "eval-loading" && (
          <div className="eval-overlay">
            <div className="eval-ring"><div className="er-o"/><div className="er-m"/><div className="er-i"/></div>
            <div className="eval-title">Evaluating your answers</div>
            <div className="eval-sub">RAG · Vector Search · tinyllama grading · MongoDB logging</div>
            <div className="eval-prog">
              <div className="eval-bar-wrap"><div className="eval-bar" style={{ width: `${evalPct}%` }}/></div>
              <div className="eval-step">{evalTxt}</div>
            </div>
          </div>
        )}

        {/* NAV */}
        <nav className="nav">
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div className="nav-mark">⚡</div>
            <div className="nav-wordmark">Quiz<em>.AI</em></div>
          </div>
          <div className="nav-tag">Ollama · {model}</div>
          <div className="nav-r">
            <div className={`o-status ${ollamaStatus}`}>
              <div className="os-dot"/>
              {ollamaStatus === "checking" ? "Checking…" :
               ollamaStatus === "connected" ? "Ollama connected" : "Ollama offline"}
            </div>
            {userName && <div className="nav-user">👤 {userName}</div>}
            <div className="nav-steps">
              {stageLabels.map((l, i) => (
                <div key={l} className={`nstep ${i < si ? "done" : i === si ? "on" : ""}`}>{l}</div>
              ))}
            </div>
          </div>
        </nav>

        {/* TECH BANNER */}
        <div className="tech-banner">
          <span className="tb-lbl">Stack</span><div className="tb-sep"/>
          {TECH_CHIPS.map(t => (
            <span key={t.l} className="tc" style={{ color: t.c, borderColor: t.c + "40", background: t.b }}>{t.l}</span>
          ))}
        </div>

        <div className="wrap">

          {/* ── NAME ── */}
          {stage === "name" && (
            <div className="hero au">
              <div className="hero-eyebrow"><span className="eyebrow-dot"/>RAG · Embeddings · Ollama · tinyllama · MongoDB</div>
              <h1 className="hero-h">
                <span className="l1">Learn smarter with</span>
                <span className="l2">local AI quizzing</span>
              </h1>
              <p className="hero-p au1">
                Upload any text document. The RAG pipeline chunks it, builds TF-IDF embeddings, indexes them in a vector store, retrieves relevant passages, then calls your local model via Ollama to generate and grade questions — 100% private, no cloud API.
              </p>

              {ollamaStatus === "disconnected" && (
                <div className="ollama-card au2" style={{ maxWidth: 400, width: "100%", marginBottom: 18 }}>
                  <div className="oc-title">⚙ Ollama Setup Required</div>
                  <div className="oc-code"><span># 1. Install from ollama.com, then:</span>OLLAMA_ORIGINS=* ollama serve</div>
                  <div className="oc-code"><span># 2. Pull tinyllama:</span>ollama pull tinyllama</div>
                  <div className="oc-note">OLLAMA_ORIGINS=* allows browser requests. Refresh after starting.</div>
                  <button className="btn btn-ghost btn-sm" style={{ marginTop: 10 }} onClick={async () => {
                    setOS("checking");
                    const { ok, models } = await checkOllama();
                    setOS(ok ? "connected" : "disconnected");
                    if (ok) { setOM(models); msg("Ollama connected!", "✅"); }
                    else msg("Still offline — check terminal", "❌");
                  }}>🔄 Retry Connection</button>
                </div>
              )}

              <div className="name-card au2">
                <label className="nc-lbl" htmlFor="uname">Your name to get started</label>
                <input id="uname" className="nc-input" placeholder="e.g. Arjun Sharma"
                  value={userName} onChange={e => setUser(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && userName.trim() && setStage("upload")} autoFocus/>
                <button className="btn btn-prime" style={{ width: "100%", justifyContent: "center", fontSize: 15 }}
                  onClick={() => setStage("upload")} disabled={!userName.trim()}>
                  Begin your journey →
                </button>
              </div>

              <div className="feat-grid au3">
                {[
                  ["🔪","Chunking"],["🔢","TF-IDF"],["🔍","Vector Search"],["🦙","Ollama LLM"],
                  ["🤖","tinyllama"],["🔒","100% Local"],["📊","Recharts"],["💡","AI Reco"],
                ].map(([ico, txt]) => (
                  <div key={txt} className="feat-item"><span className="feat-ico">{ico}</span><span className="feat-txt">{txt}</span></div>
                ))}
              </div>
            </div>
          )}

          {/* ── UPLOAD ── */}
          {stage === "upload" && (
            <div className="au">
              <div className="up-header">
                <div className="up-greeting">Welcome, {userName}</div>
                <h2 className="up-title">Upload your document</h2>
                <p className="up-sub">Text files (.txt, .md, .csv, .json) or paste directly. RAG pipeline will chunk, embed, and index via tinyllama.</p>
              </div>

              {ollamaStatus === "disconnected" && (
                <div className="ollama-card">
                  <div className="oc-title">⚠ Ollama Not Running</div>
                  <div className="oc-code"><span># Start with CORS enabled:</span>OLLAMA_ORIGINS=* ollama serve</div>
                  <button className="btn btn-ghost btn-sm" style={{ marginTop: 10 }} onClick={async () => {
                    setOS("checking");
                    const { ok, models } = await checkOllama();
                    setOS(ok ? "connected" : "disconnected");
                    if (ok) { setOM(models); msg("Connected!", "✅"); }
                    else msg("Still offline", "❌");
                  }}>🔄 Retry</button>
                </div>
              )}

              {ollamaModels.length > 1 && (
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                  <span className="cfg-lbl">Model</span>
                  <select className="cfg-sel" style={{ flex: 1, maxWidth: 240 }} value={model} onChange={e => setModel(e.target.value)}>
                    {ollamaModels.map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
              )}

              {ragInfo && (
                <div className="rag-panel au">
                  <div className="rag-title"><span className="rdot"/>RAG Pipeline — Active</div>
                  <div className="rag-grid">
                    <div className="rag-stat"><div className="rag-n">{ragInfo.chunks}</div><div className="rag-l">Chunks</div></div>
                    <div className="rag-stat"><div className="rag-n">{ragInfo.vocab.toLocaleString()}</div><div className="rag-l">Vocab</div></div>
                    <div className="rag-stat"><div className="rag-n">{vs.size}</div><div className="rag-l">Vectors</div></div>
                    <div className="rag-stat"><div className="rag-n">{MongoDB.sessions.length}</div><div className="rag-l">Sessions</div></div>
                  </div>
                </div>
              )}

              {fileName && (
                <div className="file-pill au">
                  <div className="fp-icon">📝</div>
                  <div className="fp-info">
                    <div className="fp-name">{fileName}</div>
                    <div className="fp-meta">{(charCount / 1024).toFixed(1)} KB · {wordCount.toLocaleString()} words · {sentCount} sentences</div>
                  </div>
                  <button className="fp-rm" onClick={() => { setDocText(""); setFile(""); setTopics([]); setRagInfo(null); }}>✕</button>
                </div>
              )}

              {activeText.length > 50 && (
                <div className="doc-stats au">
                  <div className="ds"><div className="ds-n">{wordCount.toLocaleString()}</div><div className="ds-l">Words</div></div>
                  <div className="ds"><div className="ds-n">{sentCount}</div><div className="ds-l">Sentences</div></div>
                  <div className="ds"><div className="ds-n">{paraCount}</div><div className="ds-l">Paragraphs</div></div>
                  <div className="ds"><div className="ds-n">{(charCount / 1024).toFixed(1)}k</div><div className="ds-l">Chars</div></div>
                </div>
              )}

              <div className={`dropzone ${drag ? "over" : ""}`}
                onClick={() => fileRef.current?.click()}
                onDragOver={e => { e.preventDefault(); setDrag(true); }}
                onDragLeave={() => setDrag(false)}
                onDrop={e => { e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0]); }}>
                <div className="dz-icon">{drag ? "🎯" : "📂"}</div>
                <div className="dz-title">{drag ? "Drop it!" : "Drop a text file here"}</div>
                <div className="dz-sub">Text files only — tinyllama processes text, not images/PDFs</div>
                <div className="dz-chips">
                  {[".txt", ".md", ".csv", ".json", ".log", ".py", ".js", ".html"].map(f => (
                    <span key={f} className="dz-chip">{f}</span>
                  ))}
                </div>
              </div>
              <input ref={fileRef} type="file"
                accept=".txt,.md,.csv,.json,.log,.py,.js,.ts,.html,.xml,.yaml,.yml"
                style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])}/>

              <div className="or-bar">or paste text directly</div>
              <textarea className="paste-ta" placeholder="Paste lecture notes, articles, study material…"
                value={paste} onChange={e => { setPaste(e.target.value); setTopics([]); }} rows={7}/>

              {topics.length > 0 && (
                <div className="topic-card au">
                  <div className="tc-hd">RAG — Detected Topics (tinyllama)</div>
                  <div className="tc-tags">{topics.map(t => <div key={t} className="tc-tag">◈ {t}</div>)}</div>
                </div>
              )}

              <div className="sec-div"/>

              <div className="cfg-sec">
                <div className="cfg-title">Quiz Configuration</div>

                {/* tinyllama tip */}
                <div style={{ fontFamily: "'DM Mono',monospace", fontSize: 10, color: "var(--gold)", background: "rgba(245,158,11,.06)", border: "1px solid rgba(245,158,11,.2)", borderRadius: 10, padding: "8px 14px", marginBottom: 14 }}>
                  💡 Tip for tinyllama: use MCQ-only + fewer questions (3–5) for best results. Mixed types may produce inconsistent JSON.
                </div>

                <div className="cfg">
                  <div className="cfg-block">
                    <label className="cfg-lbl">Questions</label>
                    <select className="cfg-sel" value={numQ} onChange={e => setNumQ(e.target.value)}>
                      {["3", "5", "7", "10"].map(n => <option key={n} value={n}>{n} Questions</option>)}
                    </select>
                  </div>
                  <div className="cfg-block">
                    <label className="cfg-lbl">Question Type</label>
                    <select className="cfg-sel" value={qType} onChange={e => setQType(e.target.value)}>
                      <option value="mcq">MCQ Only (recommended)</option>
                      <option value="mixed">Mixed (MCQ + Short)</option>
                      <option value="short">Short Answer Only</option>
                    </select>
                  </div>
                  <div className="cfg-block">
                    <label className="cfg-lbl">Difficulty</label>
                    <select className="cfg-sel" value={difficulty} onChange={e => setDiff(e.target.value)}>
                      <option value="easy">Easy</option>
                      <option value="medium">Medium</option>
                      <option value="hard">Hard</option>
                      <option value="mixed">Mixed</option>
                    </select>
                  </div>
                  <div className="cfg-block">
                    <label className="cfg-lbl">Timer</label>
                    <select className="cfg-sel" value={timerOn ? "on" : "off"} onChange={e => setTimerOn(e.target.value === "on")}>
                      <option value="off">No Timer</option>
                      <option value="on">Timed</option>
                    </select>
                  </div>
                  {timerOn && (
                    <div className="cfg-block">
                      <label className="cfg-lbl">Time Limit</label>
                      <select className="cfg-sel" value={timeLimit} onChange={e => setTimeL(e.target.value)}>
                        <option value="120">2 min</option>
                        <option value="300">5 min</option>
                        <option value="600">10 min</option>
                        <option value="900">15 min</option>
                      </select>
                    </div>
                  )}
                </div>
              </div>

              <div style={{ display: "flex", gap: 10, marginTop: 22, flexWrap: "wrap" }}>
                <button className="btn btn-prime" onClick={generate}
                  disabled={!activeText.trim() || ollamaStatus !== "connected"}>
                  ⚡ Generate Quiz {ollamaStatus !== "connected" ? "(Ollama offline)" : ""}
                </button>
                {activeText.length > 100 && topics.length === 0 && ollamaStatus === "connected" && (
                  <button className="btn btn-ghost btn-sm" onClick={extractTopics}>🔍 Extract Topics</button>
                )}
              </div>
            </div>
          )}

          {/* ── GEN LOADING ── */}
          {stage === "gen-loading" && (
            <div className="load-screen au">
              <div className="loader-ring"><div className="lr-o"/><div className="lr-i"/><div className="lr-d"/></div>
              <div className="load-title">Building your quiz</div>
              <div className="load-sub">RAG → Vector retrieval → tinyllama generation (local)</div>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                {GEN_STEPS.slice(0, genStep + 1).map((s, i) => (
                  <div key={i} className="ls-item" style={{ animationDelay: `${i * 0.1}s` }}>
                    <div className="ls-dot"/>{s}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── QUIZ ── */}
          {stage === "quiz" && (
            <div className="au">
              {parseWarn && <div className="parse-warn">⚠ {parseWarn}</div>}
              <div className="quiz-header">
                <div>
                  <div className="qh-greeting">▸ Good luck, {userName}!</div>
                  <div className="qh-title">Answer all questions</div>
                  <div className="qh-meta">{questions.length} questions · {model} · Flag to review</div>
                </div>
                <div className="qh-right">
                  {timerOn && <div className={`timer-w ${timer < 60 ? "urgent" : ""}`}>⏱ {fmtTime(timer)}</div>}
                  <div className="prog-chip"><strong>{answeredCount}</strong> / {questions.length} answered</div>
                </div>
              </div>

              {questions.map((q, qi) => {
                const ua     = answers[qi];
                const hasAns = ua !== undefined && ua !== "";
                const isFlag = flagged[qi];
                const conf   = confidence[qi];
                const dc     = diffColor[normDiff(q?.difficulty)] || "var(--text3)";
                return (
                  <div key={qi} id={`q${qi}`}
                    className={`q-card ${hasAns ? "answered" : ""} ${isFlag ? "flagged" : ""}`}
                    style={{ animationDelay: `${qi * 0.05}s` }}>
                    <div className="q-head">
                      <div className="q-num">{qi + 1}</div>
                      <div className="q-badge topic">{q.topic}</div>
                      <div className="q-badge" style={{ fontFamily: "'DM Mono',monospace", fontSize: 9, padding: "3px 10px", borderRadius: 20, background: `${dc}18`, border: `1px solid ${dc}30`, color: dc, textTransform: "uppercase" }}>{q.difficulty}</div>
                      <div className="q-badge type">{q.type === "mcq" ? "MCQ" : "Short"}</div>
                      {hasAns && !isFlag && <div className="q-check">Answered</div>}
                      <button className={`q-flag ${isFlag ? "on" : ""}`}
                        onClick={() => setFlagged(p => ({ ...p, [qi]: !p[qi] }))} title="Flag">🚩</button>
                    </div>
                    <div className="q-body">
                      <div className="q-text">{q.question}</div>
                      {q.type === "mcq" && Array.isArray(q.options) && (
                        <div className="opts">
                          {q.options.map((opt, oi) => (
                            <button key={oi} className={`opt ${ua === opt ? "sel" : ""}`}
                              onClick={() => setAns(p => ({ ...p, [qi]: opt }))}>
                              <span className="opt-key">{LETTERS[oi]}</span>{opt}
                            </button>
                          ))}
                        </div>
                      )}
                      {q.type === "short" && (
                        <>
                          <textarea className="short-ta" placeholder="Write your answer here…"
                            value={answers[qi] || ""} onChange={e => setAns(p => ({ ...p, [qi]: e.target.value }))} rows={4}/>
                          {q.hint && (
                            <div className="hint-row">
                              <span style={{ fontSize: 14, marginRight: 6 }}>💡</span>
                              <span className="hint-txt">Hint: {q.hint}</span>
                            </div>
                          )}
                        </>
                      )}
                      {hasAns && (
                        <div className="conf-row">
                          <span className="conf-label">Confidence:</span>
                          {[{ v: "high", l: "High 🟢", cls: "high" }, { v: "medium", l: "Medium 🟡", cls: "med" }, { v: "low", l: "Low 🔴", cls: "low" }].map(c => (
                            <button key={c.v} className={`conf-btn ${c.cls} ${conf === c.v ? "on" : ""}`}
                              onClick={() => setConf(p => ({ ...p, [qi]: c.v }))}>{c.l}</button>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
              <div style={{ height: 16 }}/>
            </div>
          )}

          {/* ── RESULTS ── */}
          {stage === "results" && (
            <div className="au">
              <div className="results-header">
                <div className="rank-shell" style={{ color: rank.color, "--glow": rank.glow }}>
                  <div className="rank-glow"/>
                  <div className="rank-letter" style={{ color: rank.color }}>{rank.rank}</div>
                  <div className="rank-pct">{pct}%</div>
                </div>
                <span className="res-emoji">{rank.emoji}</span>
                <div className="res-name">🎓 {userName}</div>
                <div className="res-rank-label" style={{ color: rank.color }}>{rank.label}</div>
                <div className="res-pts">{totalPts} / {maxPts} pts · Rank <strong>{rank.rank}</strong></div>
              </div>

              <div className="stat-grid">
                <div className="stat"><div className="stat-n g">{nCorr}</div><div className="stat-l">Correct</div></div>
                <div className="stat"><div className="stat-n y">{nPart}</div><div className="stat-l">Partial</div></div>
                <div className="stat"><div className="stat-n r">{nWrong}</div><div className="stat-l">Wrong</div></div>
                <div className="stat"><div className="stat-n p">{pct}%</div><div className="stat-l">Score</div></div>
                <div className="stat"><div className="stat-n b">{flaggedCount}</div><div className="stat-l">Flagged</div></div>
              </div>

              {/* Charts */}
              <div className="section-hd">Performance Analytics</div>
              <div className="chart-grid" style={{ marginBottom: 28 }}>
                {radarData.length > 1 && (
                  <div className="chart-card">
                    <div className="chart-title">Topic Radar</div>
                    <ResponsiveContainer width="100%" height={210}>
                      <RadarChart data={radarData} outerRadius={76}>
                        <PolarGrid stroke="rgba(255,255,255,0.06)"/>
                        <PolarAngleAxis dataKey="topic" tick={{ fill: "#8B82B0", fontSize: 10 }}/>
                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: "#3D3860", fontSize: 8 }}/>
                        <Radar name="Score" dataKey="score" stroke="#F97316" fill="#F97316" fillOpacity={0.2}/>
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                )}
                <div className="chart-card">
                  <div className="chart-title">Answer Breakdown</div>
                  <ResponsiveContainer width="100%" height={210}>
                    <BarChart data={barData} margin={{ top: 6, right: 6, left: -22, bottom: 6 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)"/>
                      <XAxis dataKey="name" tick={{ fill: "#8B82B0", fontSize: 11 }}/>
                      <YAxis tick={{ fill: "#3D3860", fontSize: 10 }}/>
                      <Tooltip contentStyle={{ background: "#0E0C1E", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 10, color: "#EEE9FF", fontSize: 12 }}/>
                      <Bar dataKey="value" radius={[5, 5, 0, 0]} fill="#F97316"/>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="chart-card full">
                  <div className="chart-title">Score by Difficulty</div>
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={diffData.filter(d => d.total > 0)} margin={{ top: 6, right: 6, left: -22, bottom: 6 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)"/>
                      <XAxis dataKey="diff" tick={{ fill: "#8B82B0", fontSize: 11 }}/>
                      <YAxis tick={{ fill: "#3D3860", fontSize: 10 }} domain={[0, 100]}/>
                      <Tooltip contentStyle={{ background: "#0E0C1E", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 10, color: "#EEE9FF", fontSize: 12 }} formatter={v => `${v}%`}/>
                      <Bar dataKey="pct" radius={[5, 5, 0, 0]} fill="#7C3AED"/>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Topic bars */}
              <div className="section-hd">Topic-wise Performance</div>
              {topicBreakdown().map((t, i) => (
                <div key={i} className="tb-row">
                  <div className="tb-name">{t.topic}</div>
                  <div className="tb-wrap">
                    <div className="tb-fill" style={{
                      width: `${t.pct}%`,
                      background: t.pct >= 70 ? "linear-gradient(90deg,#059669,#10B981)" : t.pct >= 50 ? "linear-gradient(90deg,#D97706,#F59E0B)" : "linear-gradient(90deg,#DC2626,#F87171)",
                      animationDuration: `${0.8 + i * 0.12}s`,
                    }}/>
                  </div>
                  <div className="tb-pct">{t.pct}%</div>
                </div>
              ))}

              {summary && (
                <div className="ai-card" style={{ marginTop: 28 }}>
                  <div className="ai-lbl">AI Performance Summary · {model} (Local)</div>
                  <div className="ai-txt">{summary}</div>
                </div>
              )}

              {recos.length > 0 && (
                <>
                  <div className="section-hd">AI Recommendations</div>
                  <div className="reco-grid">
                    {recos.map((r, i) => {
                      const rs = recoStyle(r.type);
                      const ps = prioStyle(r.priority);
                      return (
                        <div key={i} className="reco-card">
                          <div className="reco-type" style={{ color: rs.c }}>● {r.type}</div>
                          <div className="reco-topic">{r.topic}</div>
                          <div className="reco-action">{r.action}</div>
                          <div className="reco-prio" style={{ color: ps.c, background: ps.b }}>{r.priority} priority</div>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}

              <div className="export-row">
                <div className="er-lbl">📤 Download results — RAG stats, model info & recommendations</div>
                <button className="btn btn-ghost btn-sm" onClick={exportResults}>Export .txt</button>
              </div>

              <div className="section-hd">Detailed Question Review</div>
              {questions.map((q, i) => {
                const fb   = feedbacks[i];
                const ua   = answers[i];
                const conf = confidence[i];
                return (
                  <div key={i} className={`rev-card ${fb?.status || ""}`} style={{ animationDelay: `${i * 0.04}s` }}>
                    <div className="rev-pills">
                      <span className="rpill s">{q.topic}</span>
                      {fb && <span className={`rpill ${fb.status === "correct" ? "c" : fb.status === "partial" ? "p" : "i"}`}>
                        {fb.status === "correct" ? "✓ Correct" : fb.status === "partial" ? "◑ Partial" : "✗ Incorrect"}
                      </span>}
                      {fb && <span className="rpill s">{fb.score}/10</span>}
                      <span style={{ fontFamily: "'DM Mono',monospace", fontSize: 9, padding: "3px 9px", borderRadius: 20, background: `${diffColor[normDiff(q?.difficulty)]}18`, color: diffColor[normDiff(q?.difficulty)] }}>{q.difficulty}</span>
                      {flagged[i] && <span style={{ fontFamily: "'DM Mono',monospace", fontSize: 9, padding: "3px 9px", borderRadius: 20, background: "rgba(245,158,11,.07)", color: "var(--gold)" }}>🚩 Flagged</span>}
                      {conf && <span style={{ fontFamily: "'DM Mono',monospace", fontSize: 9, color: "var(--text3)" }}>conf: {conf}</span>}
                    </div>
                    <div className="rev-q">Q{i + 1}. {q.question}</div>
                    {ua && <div className="rev-ans">Your answer: <em>{ua}</em></div>}
                    {q.type === "mcq" && Array.isArray(q.options) && (
                      <div className="opts" style={{ marginTop: 10 }}>
                        {q.options.map((opt, oi) => {
                          let cls = "opt";
                          if (opt === q.answer) cls += " corr";
                          else if (opt === ua && opt !== q.answer) cls += " wrong";
                          return <button key={oi} className={cls} disabled><span className="opt-key">{LETTERS[oi]}</span>{opt}</button>;
                        })}
                      </div>
                    )}
                    {q.type === "short" && fb?.modelAns && <div className="rev-ans" style={{ marginTop: 7 }}>Model answer: <em>{fb.modelAns}</em></div>}
                    {fb && <div className="rev-fb">{fb.text}</div>}
                  </div>
                );
              })}

              {lb.length > 0 && (
                <div className="lb-card">
                  <div className="lb-hd">🏅 Session Leaderboard</div>
                  {lb.map((e, i) => (
                    <div key={i} className={`lb-row ${e.name === userName && e.pct === pct ? "me" : ""}`}>
                      <div className={`lb-pos ${i === 0 ? "g" : i === 1 ? "s" : i === 2 ? "b" : ""}`}>
                        {i === 0 ? "👑" : i === 1 ? "🥈" : i === 2 ? "🥉" : `#${i + 1}`}
                      </div>
                      <div className="lb-name">{e.name}</div>
                      <div className="lb-score">{e.pts}/{e.max} ({e.pct}%)</div>
                      <div className="lb-badge" style={{ color: e.color, background: e.color + "18" }}>{e.rank}</div>
                    </div>
                  ))}
                </div>
              )}

              <div className="res-acts">
                <button className="btn btn-prime" onClick={restart}>↩ New Quiz</button>
                <button className="btn btn-em" onClick={retake}>↺ Retake</button>
                <button className="btn btn-ghost" onClick={exportResults}>📤 Export</button>
              </div>
            </div>
          )}

        </div>

        {/* Submit bar */}
        {stage === "quiz" && (
          <div className="sub-bar">
            <div className="sb-prog">
              <div className="sb-meta"><strong>{answeredCount}</strong> of {questions.length} answered</div>
              <div className="sb-bar"><div className="sb-fill" style={{ width: `${questions.length ? (answeredCount / questions.length) * 100 : 0}%` }}/></div>
            </div>
            {flaggedCount > 0 && <div className="sb-flags">🚩 {flaggedCount} flagged</div>}
            <div className="sb-actions">
              {flaggedCount > 0 && (
                <button className="btn btn-ghost btn-sm" onClick={() => {
                  const fi = Object.entries(flagged).find(([k, v]) => v);
                  if (fi) document.getElementById(`q${fi[0]}`)?.scrollIntoView({ behavior: "smooth" });
                }}>Jump to flagged</button>
              )}
              <button className="btn btn-prime" onClick={evaluateAll} disabled={answeredCount === 0}>
                {answeredCount === questions.length ? "Submit & Evaluate →" : `Submit (${answeredCount}/${questions.length})`}
              </button>
            </div>
          </div>
        )}

      </div>

      {toast && (
        <div className="toast">
          <span>{toast.icon}</span>
          <span>{toast.m}</span>
        </div>
      )}
    </>
  );
}
