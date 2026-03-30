import { useState, useRef, useEffect, useMemo } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, Legend
} from "recharts";

// ─────────────────────────────────────────────
//  GEMINI API  (text + vision)
// ─────────────────────────────────────────────
async function gemini(prompt, system = "", maxTokens = 2048) {
  const model = "gemini-2.5-flash";
  const key = import.meta.env.VITE_GEMINI_API_KEY?.trim();
  if (!key) throw new Error("Missing VITE_GEMINI_API_KEY in .env");
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(key)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        system_instruction: system ? { parts: [{ text: system }] } : undefined,
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: { maxOutputTokens: maxTokens, temperature: 0.75 },
      }),
    }
  );
  const data = await res.json();
  if (!res.ok) throw new Error(data?.error?.message || `HTTP ${res.status}`);
  return data?.candidates?.[0]?.content?.parts?.[0]?.text || "";
}

// ─── MLLM / Vision: send image OR PDF base64 to Gemini vision ─────────────
async function geminiVision(base64Data, mimeType, prompt) {
  const model = "gemini-2.5-flash";
  const key = import.meta.env.VITE_GEMINI_API_KEY?.trim();
  if (!key) throw new Error("Missing VITE_GEMINI_API_KEY in .env");
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(key)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{
          role: "user",
          parts: [
            { inline_data: { mime_type: mimeType, data: base64Data } },
            { text: prompt }
          ]
        }],
        generationConfig: { maxOutputTokens: 4096, temperature: 0.4 },
      }),
    }
  );
  const data = await res.json();
  if (!res.ok) throw new Error(data?.error?.message || `HTTP ${res.status}`);
  return data?.candidates?.[0]?.content?.parts?.[0]?.text || "";
}

// ─────────────────────────────────────────────
//  TEXT CHUNKING  (RAG layer 1)
// ─────────────────────────────────────────────
function chunkText(text, chunkSize = 400, overlap = 80) {
  const words = text.split(/\s+/).filter(Boolean);
  const chunks = [];
  let i = 0;
  while (i < words.length) {
    const chunk = words.slice(i, i + chunkSize).join(" ");
    chunks.push({ id: chunks.length, text: chunk, wordStart: i });
    i += chunkSize - overlap;
  }
  return chunks;
}

// ─────────────────────────────────────────────
//  EMBEDDINGS  (cosine-ready TF-IDF vectors, client-side)
// ─────────────────────────────────────────────
function buildVocab(chunks) {
  const vocab = {};
  let idx = 0;
  chunks.forEach(c => {
    c.text.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).forEach(w => {
      if (w.length > 2 && !vocab[w]) vocab[w] = idx++;
    });
  });
  return vocab;
}

function tfIdfVector(text, vocab, idfMap) {
  const words = text.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(w => w.length > 2);
  const tf = {};
  words.forEach(w => { tf[w] = (tf[w] || 0) + 1; });
  const vec = new Float32Array(Object.keys(vocab).length);
  Object.entries(tf).forEach(([w, count]) => {
    if (vocab[w] !== undefined) {
      vec[vocab[w]] = (count / words.length) * (idfMap[w] || 1);
    }
  });
  return vec;
}

function buildIdf(chunks, vocab) {
  const df = {};
  chunks.forEach(c => {
    const words = new Set(c.text.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(w => w.length > 2));
    words.forEach(w => { df[w] = (df[w] || 0) + 1; });
  });
  const idf = {};
  Object.keys(vocab).forEach(w => {
    idf[w] = Math.log((chunks.length + 1) / ((df[w] || 0) + 1)) + 1;
  });
  return idf;
}

function cosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

// ─────────────────────────────────────────────
//  VECTOR STORE  (in-memory MongoDB-like)
// ─────────────────────────────────────────────
class VectorStore {
  constructor() {
    this._collection = [];
    this._vocab = null;
    this._idf = null;
    this._dbName = "quizai_rag";
    this._collName = "chunks";
  }
  insertMany(chunks, vocab, idf) {
    this._vocab = vocab;
    this._idf = idf;
    this._collection = chunks.map(c => ({
      _id: `chunk_${c.id}`,
      chunkId: c.id,
      text: c.text,
      vector: tfIdfVector(c.text, vocab, idf),
      metadata: { wordStart: c.wordStart, length: c.text.split(" ").length }
    }));
  }
  vectorSearch(queryText, topK = 4) {
    if (!this._collection.length || !this._vocab) return [];
    const qVec = tfIdfVector(queryText, this._vocab, this._idf);
    return this._collection
      .map(doc => ({ ...doc, score: cosineSimilarity(qVec, doc.vector) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
  get size() { return this._collection.length; }
  get dbInfo() { return `${this._dbName}.${this._collName} — ${this._collection.length} documents`; }
}

const vectorStore = new VectorStore();

// ─────────────────────────────────────────────
//  OCR  — extract text from image OR PDF via Gemini Vision
// ─────────────────────────────────────────────
async function ocrFile(base64, mime) {
  const isPDF = mime === "application/pdf";
  const prompt = isPDF
    ? "Extract ALL text from this PDF document. Return plain text only, preserving the logical reading order and structure. Include all headings, paragraphs, tables (as text), and any visible text labels."
    : "Extract ALL text visible in this image. Return plain text only, preserving structure. If it's a diagram or chart, describe it then extract any text labels.";
  return await geminiVision(base64, mime, prompt);
}

// kept for backward compat
const ocrImage = ocrFile;

// ─────────────────────────────────────────────
//  AI RECOMMENDATION ENGINE
// ─────────────────────────────────────────────
async function generateRecommendations(topicBreakdown, userName, score) {
  const weakTopics   = topicBreakdown.filter(t => t.pct < 60).map(t => `${t.topic} (${t.pct}%)`).join(", ");
  const strongTopics = topicBreakdown.filter(t => t.pct >= 80).map(t => `${t.topic} (${t.pct}%)`).join(", ");
  const raw = await gemini(
    `Student: ${userName}. Score: ${score}%.\nWeak areas: ${weakTopics || "none"}.\nStrong areas: ${strongTopics || "none"}.\n\nReturn ONLY a JSON array of exactly 4 recommendations:
[{"type":"review"|"practice"|"advance"|"resource","topic":"string","action":"string (1 sentence)","priority":"high"|"medium"|"low"}]`,
    "You are a learning coach. Return only valid JSON.", 600
  );
  try {
    const arr = JSON.parse(raw.replace(/```json|```/gi, "").trim());
    return Array.isArray(arr) ? arr : [];
  } catch { return []; }
}

// ─────────────────────────────────────────────
//  HELPERS
// ─────────────────────────────────────────────
const LETTERS = ["A", "B", "C", "D"];
const fmtTime = s => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

const getRank = pct => {
  if (pct >= 95) return { rank: "S+", label: "Legendary",   color: "#FFD700", glow: "rgba(255,215,0,0.4)",   emoji: "👑" };
  if (pct >= 90) return { rank: "S",  label: "Outstanding", color: "#E879F9", glow: "rgba(232,121,249,0.4)", emoji: "⭐" };
  if (pct >= 80) return { rank: "A",  label: "Excellent",   color: "#34D399", glow: "rgba(52,211,153,0.4)",  emoji: "🏆" };
  if (pct >= 70) return { rank: "B",  label: "Good",        color: "#38BDF8", glow: "rgba(56,189,248,0.4)",  emoji: "🎯" };
  if (pct >= 60) return { rank: "C",  label: "Average",     color: "#FBBF24", glow: "rgba(251,191,36,0.4)",  emoji: "📚" };
  if (pct >= 40) return { rank: "D",  label: "Needs Work",  color: "#FB923C", glow: "rgba(251,146,60,0.4)",  emoji: "📖" };
  return               { rank: "F",  label: "Keep Trying", color: "#F87171", glow: "rgba(248,113,113,0.4)", emoji: "💪" };
};

const diffColor = { easy: "#34D399", medium: "#FBBF24", hard: "#F87171" };

const readFileAsText = f => new Promise((res, rej) => {
  const r = new FileReader(); r.onload = () => res(r.result); r.onerror = rej; r.readAsText(f);
});
const readFileAsBase64 = f => new Promise((res, rej) => {
  const r = new FileReader(); r.onload = () => res(r.result.split(",")[1]); r.onerror = rej; r.readAsDataURL(f);
});

const parseJSON = raw => {
  try { return JSON.parse(raw.replace(/```json|```/gi, "").trim()); } catch { return null; }
};

// ─── MongoDB stub ─────────────────────────────────────────────────────────
const MongoDB = {
  sessions: JSON.parse(sessionStorage.getItem("mongo_sessions") || "[]"),
  save(doc) {
    this.sessions = [doc, ...this.sessions].slice(0, 20);
    sessionStorage.setItem("mongo_sessions", JSON.stringify(this.sessions));
    console.log("[MongoDB] db.sessions.insertOne:", JSON.stringify(doc).slice(0, 120) + "…");
  },
  find(query = {}) {
    console.log("[MongoDB] db.sessions.find:", query);
    return this.sessions;
  }
};

// ─── Leaderboard ──────────────────────────────────────────────────────────
let LEADERBOARD = JSON.parse(sessionStorage.getItem("quiz_lb3") || "[]");
const saveLB = entry => {
  LEADERBOARD = [entry, ...LEADERBOARD].slice(0, 8).sort((a, b) => b.pct - a.pct);
  sessionStorage.setItem("quiz_lb3", JSON.stringify(LEADERBOARD));
};

// ─────────────────────────────────────────────
//  CSS
// ─────────────────────────────────────────────
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Playfair+Display:ital,wght@0,700;0,900;1,600;1,700&display=swap');

:root {
  --ink:#03020A; --ink1:#080714; --ink2:#0E0C1E; --ink3:#151228; --ink4:#1C1835;
  --glass:rgba(255,255,255,0.028); --glass2:rgba(255,255,255,0.055); --glass3:rgba(255,255,255,0.09);
  --rim:rgba(255,255,255,0.06); --rim2:rgba(255,255,255,0.11);
  --violet:#7C3AED; --violet2:#A78BFA; --violet3:#C4B5FD;
  --cyan:#06B6D4; --cyan2:#67E8F9; --rose:#FB7185; --gold:#F59E0B; --emerald:#10B981; --sky:#38BDF8;
  --text:#EEE9FF; --text2:#8B82B0; --text3:#3D3860;
  --r:18px; --r2:12px; --r3:8px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--ink);color:var(--text);font-family:'Syne',system-ui,sans-serif;-webkit-font-smoothing:antialiased;min-height:100vh;overflow-x:hidden;cursor:default}
::selection{background:rgba(124,58,237,0.35);color:var(--violet3)}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(124,58,237,0.25);border-radius:4px}

.bg-canvas{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.nebula{position:absolute;border-radius:50%;filter:blur(120px)}
.nb1{width:900px;height:900px;top:-300px;left:-200px;background:radial-gradient(circle,rgba(124,58,237,0.09) 0%,transparent 70%);animation:nb-drift 25s ease-in-out infinite alternate}
.nb2{width:700px;height:700px;top:30%;right:-200px;background:radial-gradient(circle,rgba(6,182,212,0.06) 0%,transparent 70%);animation:nb-drift 30s ease-in-out infinite alternate-reverse}
.nb3{width:600px;height:600px;bottom:-100px;left:20%;background:radial-gradient(circle,rgba(251,113,133,0.05) 0%,transparent 70%);animation:nb-drift 22s ease-in-out infinite alternate}
.nb4{width:400px;height:400px;top:50%;left:40%;background:radial-gradient(circle,rgba(245,158,11,0.04) 0%,transparent 70%);animation:nb-drift 18s ease-in-out infinite alternate-reverse}
@keyframes nb-drift{0%{transform:translate(0,0)scale(1)}33%{transform:translate(50px,-40px)scale(1.08)}66%{transform:translate(-30px,60px)scale(0.94)}100%{transform:translate(70px,25px)scale(1.04)}}
.stars{position:absolute;inset:0;background-image:radial-gradient(1px 1px at 10% 20%,rgba(255,255,255,0.4) 0%,transparent 100%),radial-gradient(1px 1px at 30% 70%,rgba(255,255,255,0.3) 0%,transparent 100%),radial-gradient(1.5px 1.5px at 60% 15%,rgba(200,180,255,0.5) 0%,transparent 100%),radial-gradient(1px 1px at 80% 55%,rgba(255,255,255,0.25) 0%,transparent 100%),radial-gradient(1px 1px at 45% 85%,rgba(180,220,255,0.4) 0%,transparent 100%),radial-gradient(1.5px 1.5px at 92% 30%,rgba(255,255,255,0.35) 0%,transparent 100%);animation:twinkle 8s ease-in-out infinite alternate}
@keyframes twinkle{0%{opacity:0.6}50%{opacity:1}100%{opacity:0.7}}
.scanlines{position:absolute;inset:0;pointer-events:none;z-index:1;background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.03) 3px,rgba(0,0,0,0.03) 4px)}
.grid-bg{position:absolute;inset:0;background-image:linear-gradient(rgba(124,58,237,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(124,58,237,0.03) 1px,transparent 1px);background-size:80px 80px;mask-image:radial-gradient(ellipse 80% 80% at 50% 50%,black 0%,transparent 100%)}

@keyframes slide-up{from{opacity:0;transform:translateY(32px)}to{opacity:1;transform:translateY(0)}}
@keyframes fade-in{from{opacity:0}to{opacity:1}}
@keyframes pop{0%{transform:scale(0.5);opacity:0}70%{transform:scale(1.06)}100%{transform:scale(1);opacity:1}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes spin-rev{to{transform:rotate(-360deg)}}
@keyframes pulse-glow{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.5;transform:scale(0.95)}}
@keyframes shimmer{0%{background-position:200% center}100%{background-position:-200% center}}
@keyframes slide-r{from{transform:translateX(-20px);opacity:0}to{transform:translateX(0);opacity:1}}
@keyframes bar-fill{from{width:0}}
@keyframes float{0%,100%{transform:translateY(0px)rotate(-1deg)}50%{transform:translateY(-10px)rotate(1deg)}}
@keyframes halo-spin{to{transform:rotate(360deg)}}

.au{animation:slide-up 0.6s cubic-bezier(0.16,1,0.3,1) both}
.au2{animation:slide-up 0.6s cubic-bezier(0.16,1,0.3,1) 0.1s both}
.au3{animation:slide-up 0.6s cubic-bezier(0.16,1,0.3,1) 0.2s both}
.au4{animation:slide-up 0.6s cubic-bezier(0.16,1,0.3,1) 0.3s both}
.pop{animation:pop 0.7s cubic-bezier(0.34,1.56,0.64,1) both}
.fi{animation:fade-in 0.5s ease both}

.page{position:relative;z-index:1;min-height:100vh}

/* NAV */
.nav{display:flex;align-items:center;gap:14px;padding:14px 40px;background:rgba(3,2,10,0.7);backdrop-filter:blur(32px) saturate(1.5);border-bottom:1px solid var(--rim);position:sticky;top:0;z-index:200}
.nav-logo{display:flex;align-items:center;gap:12px}
.nav-mark{position:relative;width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#7C3AED 0%,#06B6D4 100%);display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;box-shadow:0 0 0 1px rgba(124,58,237,0.4),0 0 24px rgba(124,58,237,0.3),0 0 60px rgba(6,182,212,0.1)}
.nav-mark::before{content:'';position:absolute;inset:-3px;border-radius:15px;background:conic-gradient(from 0deg,transparent 0%,rgba(124,58,237,0.4) 25%,transparent 50%);animation:halo-spin 4s linear infinite;z-index:-1}
.nav-wordmark{font-size:17px;font-weight:800;letter-spacing:-0.8px}
.nav-wordmark em{font-style:normal;background:linear-gradient(90deg,var(--violet2),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.nav-tag{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:0.12em;padding:3px 10px;border-radius:30px;text-transform:uppercase;background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.2);color:var(--violet2)}
.nav-r{margin-left:auto;display:flex;align-items:center;gap:12px}
.nav-user-chip{display:flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;font-size:11px;color:var(--cyan2);padding:5px 14px;border-radius:30px;background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.2)}
.nav-user-avatar{width:20px;height:20px;border-radius:6px;background:linear-gradient(135deg,var(--violet),var(--cyan));display:flex;align-items:center;justify-content:center;font-size:10px}
.nav-steps{display:flex;align-items:center;gap:2px}
.nstep{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:0.08em;text-transform:uppercase;padding:5px 14px;border-radius:30px;border:1px solid transparent;color:var(--text3);transition:all 0.3s}
.nstep::after{content:'›';margin-left:6px;opacity:0.3}
.nstep:last-child::after{display:none}
.nstep.on{color:var(--violet3);border-color:rgba(124,58,237,0.3);background:rgba(124,58,237,0.1)}
.nstep.done{color:var(--emerald);border-color:rgba(16,185,129,0.25)}

/* TECH STACK BANNER */
.tech-banner{display:flex;align-items:center;gap:8px;padding:8px 40px;background:rgba(8,7,20,0.6);border-bottom:1px solid var(--rim);overflow-x:auto;flex-wrap:nowrap}
.tech-banner::-webkit-scrollbar{height:2px}
.tb-label{font-family:'DM Mono',monospace;font-size:8px;color:var(--text3);letter-spacing:0.16em;text-transform:uppercase;white-space:nowrap;flex-shrink:0}
.tb-sep{width:1px;height:14px;background:var(--rim2);flex-shrink:0}
.tech-chip{font-family:'DM Mono',monospace;font-size:8px;font-weight:500;padding:3px 9px;border-radius:4px;white-space:nowrap;flex-shrink:0;border:1px solid;letter-spacing:0.06em;cursor:default;transition:all 0.2s}
.tech-chip:hover{filter:brightness(1.3);transform:translateY(-1px)}

.wrap{max-width:840px;margin:0 auto;padding:60px 28px 130px}

/* BUTTONS */
.btn{padding:13px 26px;border-radius:var(--r2);font-family:'Syne',sans-serif;font-weight:700;font-size:14px;cursor:pointer;border:none;outline:none;display:inline-flex;align-items:center;gap:9px;transition:all 0.2s cubic-bezier(0.16,1,0.3,1);position:relative;overflow:hidden;letter-spacing:0.01em}
.btn-prime{background:linear-gradient(135deg,#7C3AED 0%,#06B6D4 100%);color:#fff;box-shadow:0 4px 24px rgba(124,58,237,0.35),0 1px 0 rgba(255,255,255,0.1) inset}
.btn-prime:hover:not(:disabled){transform:translateY(-2px) scale(1.01);box-shadow:0 8px 40px rgba(124,58,237,0.5)}
.btn-prime:disabled{opacity:0.3;cursor:not-allowed}
.btn-emerald{background:linear-gradient(135deg,#059669,#0891B2);color:#fff;box-shadow:0 4px 20px rgba(5,150,105,0.3)}
.btn-emerald:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 8px 32px rgba(5,150,105,0.45)}
.btn-ghost{background:var(--glass2);color:var(--text2);border:1px solid var(--rim2);backdrop-filter:blur(8px)}
.btn-ghost:hover{background:var(--glass3);color:var(--text);border-color:rgba(255,255,255,0.15)}
.btn-sm{padding:8px 16px;font-size:12px}

/* HERO */
.hero{display:flex;flex-direction:column;align-items:center;text-align:center;padding:80px 0 60px}
.hero-eyebrow{display:inline-flex;align-items:center;gap:10px;font-family:'DM Mono',monospace;font-size:10px;letter-spacing:0.2em;text-transform:uppercase;color:var(--cyan2);padding:8px 20px;border-radius:40px;margin-bottom:36px;background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.2);animation:slide-up 0.5s ease both}
.hero-eyebrow-dot{width:6px;height:6px;border-radius:50%;background:var(--cyan);animation:pulse-glow 2s ease infinite;box-shadow:0 0 8px var(--cyan)}
.hero-h{font-size:clamp(52px,8vw,92px);font-weight:900;line-height:0.9;letter-spacing:-4px;margin-bottom:24px;animation:slide-up 0.6s cubic-bezier(0.16,1,0.3,1) 0.08s both}
.hero-h .l1{display:block;color:var(--text)}
.hero-h .l2{display:block;font-family:'Playfair Display',serif;font-style:italic;font-weight:700;font-size:clamp(58px,9vw,100px);background:linear-gradient(135deg,var(--violet2) 0%,var(--cyan2) 40%,#F59E0B 80%,var(--rose) 100%);background-size:200% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:slide-up 0.6s cubic-bezier(0.16,1,0.3,1) 0.08s both,shimmer 4s linear 1s infinite;letter-spacing:-3px}
.hero-p{font-size:16px;line-height:1.8;color:var(--text2);max-width:480px;margin-bottom:56px;font-weight:400;animation:slide-up 0.6s cubic-bezier(0.16,1,0.3,1) 0.16s both}
.name-card{width:100%;max-width:420px;background:rgba(14,12,30,0.7);border:1px solid var(--rim2);border-radius:24px;padding:40px 36px;backdrop-filter:blur(40px);box-shadow:0 40px 100px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.05);animation:slide-up 0.7s cubic-bezier(0.16,1,0.3,1) 0.24s both;position:relative;overflow:hidden}
.name-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(124,58,237,0.5),rgba(6,182,212,0.5),transparent)}
.nc-label{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:0.18em;text-transform:uppercase;color:var(--text3);margin-bottom:12px;display:block}
.nc-input{width:100%;padding:16px 20px;margin-bottom:20px;border:1px solid var(--rim2);border-radius:14px;background:rgba(0,0,0,0.35);color:var(--text);font-family:'Syne',sans-serif;font-size:20px;font-weight:700;outline:none;transition:all 0.25s;text-align:center;letter-spacing:-0.3px}
.nc-input:focus{border-color:rgba(124,58,237,0.5);box-shadow:0 0 0 4px rgba(124,58,237,0.1);background:rgba(0,0,0,0.5)}
.nc-input::placeholder{color:var(--text3);font-weight:400;font-size:16px}
.feat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:40px;max-width:680px;animation:slide-up 0.7s cubic-bezier(0.16,1,0.3,1) 0.32s both}
.feat-item{display:flex;flex-direction:column;align-items:center;gap:7px;padding:16px 12px;border-radius:16px;border:1px solid var(--rim);background:var(--glass);backdrop-filter:blur(8px);transition:all 0.2s}
.feat-item:hover{border-color:rgba(124,58,237,0.25);background:rgba(124,58,237,0.05);transform:translateY(-2px)}
.feat-ico{font-size:22px;line-height:1}
.feat-txt{font-family:'DM Mono',monospace;font-size:9px;letter-spacing:0.06em;text-transform:uppercase;color:var(--text3);text-align:center}

/* UPLOAD */
.up-header{margin-bottom:36px}
.up-greeting{font-family:'DM Mono',monospace;font-size:11px;color:var(--cyan);letter-spacing:0.08em;margin-bottom:8px;display:flex;align-items:center;gap:10px}
.up-greeting::before{content:'';display:block;width:24px;height:1px;background:var(--cyan)}
.up-title{font-size:36px;font-weight:800;letter-spacing:-1.5px;margin-bottom:8px}
.up-sub{font-size:14px;color:var(--text2);line-height:1.7}

/* RAG STATUS PANEL */
.rag-panel{margin-bottom:20px;padding:16px 20px;border-radius:16px;border:1px solid rgba(124,58,237,0.18);background:rgba(124,58,237,0.03);animation:slide-up 0.4s ease both}
.rag-title{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:0.16em;color:var(--violet2);margin-bottom:12px;display:flex;align-items:center;gap:8px}
.rag-title .rdot{width:6px;height:6px;border-radius:50%;background:var(--violet2);box-shadow:0 0 6px var(--violet2);animation:pulse-glow 1.5s ease infinite}
.rag-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.rag-stat{background:rgba(0,0,0,0.25);border-radius:10px;padding:10px;text-align:center;border:1px solid var(--rim)}
.rag-n{font-family:'DM Mono',monospace;font-size:18px;font-weight:500;color:var(--violet2);line-height:1}
.rag-l{font-family:'DM Mono',monospace;font-size:8px;color:var(--text3);text-transform:uppercase;letter-spacing:0.1em;margin-top:4px}

.dropzone{border:1px dashed rgba(124,58,237,0.25);border-radius:20px;padding:56px 36px;text-align:center;cursor:pointer;transition:all 0.3s cubic-bezier(0.16,1,0.3,1);position:relative;overflow:hidden;background:linear-gradient(135deg,rgba(124,58,237,0.02),rgba(6,182,212,0.01))}
.dropzone:hover,.dropzone.over{border-color:rgba(124,58,237,0.5);border-style:solid;box-shadow:0 0 0 1px rgba(124,58,237,0.15),0 24px 80px rgba(0,0,0,0.3);transform:translateY(-4px)}
.dz-center{position:relative;z-index:1}
.dz-icon-ring{width:72px;height:72px;margin:0 auto 20px;border-radius:22px;background:linear-gradient(135deg,rgba(124,58,237,0.15),rgba(6,182,212,0.1));border:1px solid rgba(124,58,237,0.2);display:flex;align-items:center;justify-content:center;font-size:30px;transition:all 0.3s}
.dropzone:hover .dz-icon-ring{transform:scale(1.1) rotate(-5deg);border-color:rgba(124,58,237,0.45);box-shadow:0 0 40px rgba(124,58,237,0.25)}
.dz-title{font-size:18px;font-weight:700;margin-bottom:4px}
.dz-sub{font-family:'DM Mono',monospace;font-size:11px;color:var(--text3);margin-bottom:14px}
.dz-chips{display:flex;gap:6px;justify-content:center;flex-wrap:wrap}
.dz-chip{font-family:'DM Mono',monospace;font-size:9px;padding:3px 10px;border-radius:20px;background:rgba(124,58,237,0.07);border:1px solid var(--rim);color:var(--text3);letter-spacing:0.06em}
/* PDF chip gets a distinct red tint */
.dz-chip.pdf{background:rgba(239,68,68,0.08);border-color:rgba(239,68,68,0.2);color:#FCA5A5}

.file-pill{display:flex;align-items:center;gap:14px;padding:14px 18px;border-radius:16px;border:1px solid rgba(124,58,237,0.22);background:rgba(124,58,237,0.05);margin-bottom:16px;animation:slide-r 0.35s ease both}
.fp-icon{width:46px;height:46px;border-radius:14px;background:linear-gradient(135deg,rgba(124,58,237,0.3),rgba(6,182,212,0.15));display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0}
.fp-info{flex:1}
.fp-name{font-weight:700;font-size:14px}
.fp-meta{font-family:'DM Mono',monospace;font-size:10px;color:var(--text3);margin-top:3px}
.fp-rm{width:30px;height:30px;border-radius:8px;border:1px solid var(--rim);background:transparent;color:var(--text3);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:13px;transition:all 0.2s}
.fp-rm:hover{border-color:#F87171;color:#F87171;background:rgba(248,113,113,0.08)}

.doc-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--rim);border:1px solid var(--rim);border-radius:16px;overflow:hidden;margin-bottom:24px;animation:slide-up 0.4s ease both}
.ds{background:var(--ink2);padding:18px 14px;text-align:center;transition:background 0.2s}
.ds:hover{background:var(--ink3)}
.ds-n{font-size:24px;font-weight:800;color:var(--violet2);letter-spacing:-0.5px;line-height:1}
.ds-l{font-family:'DM Mono',monospace;font-size:8px;color:var(--text3);text-transform:uppercase;letter-spacing:0.1em;margin-top:5px}

.or-bar{display:flex;align-items:center;gap:14px;margin:20px 0;font-family:'DM Mono',monospace;font-size:10px;color:var(--text3);letter-spacing:0.08em}
.or-bar::before,.or-bar::after{content:'';flex:1;height:1px;background:var(--rim)}
.paste-ta{width:100%;min-height:150px;padding:18px 20px;border:1px solid var(--rim);border-radius:16px;background:var(--ink1);color:var(--text);resize:vertical;font-family:'Syne',sans-serif;font-size:14px;line-height:1.75;outline:none;transition:all 0.25s}
.paste-ta:focus{border-color:rgba(124,58,237,0.4);box-shadow:0 0 0 3px rgba(124,58,237,0.08);background:var(--ink2)}
.paste-ta::placeholder{color:var(--text3)}

.topic-card{margin-top:24px;padding:20px 24px;border-radius:16px;border:1px solid rgba(6,182,212,0.18);background:rgba(6,182,212,0.03);animation:slide-up 0.4s ease both}
.tc-hd{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:0.16em;color:var(--cyan);margin-bottom:14px;display:flex;align-items:center;gap:10px}
.tc-hd::after{content:'';flex:1;height:1px;background:rgba(6,182,212,0.2)}
.tc-tags{display:flex;flex-wrap:wrap;gap:8px}
.tc-tag{font-size:12px;font-weight:600;padding:5px 14px;border-radius:30px;background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.18);color:var(--cyan2);transition:all 0.2s}
.tc-tag:hover{background:rgba(6,182,212,0.12)}

.cfg-section{margin-top:32px}
.cfg-title{font-size:18px;font-weight:800;letter-spacing:-0.4px;margin-bottom:18px;display:flex;align-items:center;gap:10px}
.cfg-title::after{content:'';flex:1;height:1px;background:var(--rim)}
.cfg{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.cfg-block{display:flex;flex-direction:column;gap:8px}
.cfg-lbl{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:0.14em;color:var(--text3)}
.cfg-sel{width:100%;padding:12px 16px;border:1px solid var(--rim);border-radius:12px;background:var(--ink2);color:var(--text);font-family:'Syne',sans-serif;font-size:13px;font-weight:600;outline:none;cursor:pointer;transition:all 0.2s;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%233D3860' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 14px center}
.cfg-sel:focus{border-color:rgba(124,58,237,0.4);box-shadow:0 0 0 3px rgba(124,58,237,0.08)}
.sec-div{height:1px;background:var(--rim);margin:28px 0}

/* OCR / PDF notice */
.ocr-notice{margin-top:10px;padding:10px 14px;border-radius:10px;background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.15);font-family:'DM Mono',monospace;font-size:10px;color:rgba(16,185,129,0.8);display:flex;align-items:center;gap:8px}
.pdf-notice{background:rgba(239,68,68,0.05);border-color:rgba(239,68,68,0.2);color:#FCA5A5}

/* LOADING */
.load-screen{text-align:center;padding:100px 24px}
.loader-ring{width:72px;height:72px;margin:0 auto 32px;position:relative}
.lr-outer{position:absolute;inset:0;border:2px solid transparent;border-top-color:var(--violet);border-right-color:rgba(124,58,237,0.4);border-radius:50%;animation:spin 0.9s linear infinite;box-shadow:0 0 30px rgba(124,58,237,0.25)}
.lr-inner{position:absolute;inset:10px;border:1.5px solid transparent;border-bottom-color:var(--cyan);border-left-color:rgba(6,182,212,0.35);border-radius:50%;animation:spin-rev 0.6s linear infinite}
.lr-dot{position:absolute;top:50%;left:50%;width:8px;height:8px;border-radius:50%;background:var(--violet2);transform:translate(-50%,-50%);box-shadow:0 0 16px var(--violet);animation:pulse-glow 1s ease infinite}
.load-title{font-size:30px;font-weight:800;letter-spacing:-1px;margin-bottom:8px}
.load-sub{font-size:14px;color:var(--text2);margin-bottom:36px}
.load-steps{display:flex;flex-direction:column;gap:10px;align-items:center}
.ls-item{display:flex;align-items:center;gap:12px;font-family:'DM Mono',monospace;font-size:11px;color:var(--text3);animation:slide-up 0.4s ease both}
.ls-dot{width:5px;height:5px;border-radius:50%;background:var(--violet);box-shadow:0 0 8px var(--violet);animation:pulse-glow 1.2s ease infinite}

/* QUIZ */
.quiz-header{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:36px;gap:20px;flex-wrap:wrap}
.qh-greeting{font-family:'DM Mono',monospace;font-size:10px;color:var(--cyan);letter-spacing:0.08em;margin-bottom:6px;display:flex;align-items:center;gap:8px}
.qh-greeting::before{content:'▸';opacity:0.6}
.qh-title{font-size:32px;font-weight:800;letter-spacing:-1.2px;margin-bottom:4px}
.qh-meta{font-size:13px;color:var(--text3)}
.qh-right{display:flex;flex-direction:column;align-items:flex-end;gap:8px}
.timer-widget{font-family:'DM Mono',monospace;font-size:24px;font-weight:500;color:var(--gold);padding:10px 20px;border-radius:14px;background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.2);display:flex;align-items:center;gap:10px;letter-spacing:0.04em}
.timer-widget.urgent{color:#F87171;background:rgba(248,113,113,0.07);border-color:rgba(248,113,113,0.25);animation:pulse-glow 1s ease infinite}
.progress-chip{font-family:'DM Mono',monospace;font-size:10px;padding:6px 16px;border-radius:30px;color:var(--text2);background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.2)}
.progress-chip strong{color:var(--violet2)}

.q-card{border-radius:20px;border:1px solid var(--rim);background:var(--ink1);overflow:hidden;margin-bottom:16px;transition:all 0.25s cubic-bezier(0.16,1,0.3,1);animation:slide-up 0.5s cubic-bezier(0.16,1,0.3,1) both;position:relative}
.q-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(124,58,237,0.3),transparent);opacity:0;transition:opacity 0.3s}
.q-card:hover::before{opacity:1}
.q-card:hover{border-color:var(--rim2);box-shadow:0 8px 40px rgba(0,0,0,0.25)}
.q-card.answered{border-color:rgba(124,58,237,0.2)}
.q-card.answered::before{opacity:1;background:linear-gradient(90deg,transparent,rgba(124,58,237,0.4),rgba(6,182,212,0.3),transparent)}
.q-card.flagged{border-color:rgba(245,158,11,0.3)}
.q-card.flagged::before{opacity:1;background:linear-gradient(90deg,transparent,rgba(245,158,11,0.4),transparent)}
.q-head{display:flex;align-items:center;gap:10px;padding:16px 20px 13px;border-bottom:1px solid var(--rim);background:rgba(0,0,0,0.2)}
.q-num{width:34px;height:34px;border-radius:10px;flex-shrink:0;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,rgba(124,58,237,0.25),rgba(6,182,212,0.1));border:1px solid rgba(124,58,237,0.2);font-size:13px;font-weight:800;color:var(--violet2)}
.q-badge{font-family:'DM Mono',monospace;font-size:9px;padding:3px 10px;border-radius:20px;letter-spacing:0.07em;text-transform:uppercase}
.q-badge.topic{background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.15);color:var(--cyan)}
.q-badge.type{background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.15);color:var(--gold)}
.q-check{margin-left:auto;font-family:'DM Mono',monospace;font-size:9px;color:var(--emerald);display:flex;align-items:center;gap:6px}
.q-check::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--emerald);box-shadow:0 0 8px var(--emerald)}
.q-flag{width:30px;height:30px;border-radius:8px;border:1px solid var(--rim);background:transparent;color:var(--text3);cursor:pointer;font-size:14px;display:flex;align-items:center;justify-content:center;transition:all 0.2s}
.q-flag:hover{border-color:rgba(245,158,11,0.4);color:var(--gold);background:rgba(245,158,11,0.06)}
.q-flag.on{border-color:rgba(245,158,11,0.4);background:rgba(245,158,11,0.08);color:var(--gold)}
.q-body{padding:22px 20px}
.q-text{font-size:17px;font-weight:700;line-height:1.55;margin-bottom:22px;letter-spacing:-0.2px}

.opts{display:flex;flex-direction:column;gap:8px}
.opt{display:flex;align-items:flex-start;gap:13px;padding:13px 16px;border-radius:12px;border:1px solid var(--rim);background:var(--ink2);cursor:pointer;font-size:14px;line-height:1.55;font-weight:500;transition:all 0.15s;color:var(--text);width:100%;font-family:'Syne',sans-serif;text-align:left}
.opt:hover:not(:disabled){border-color:rgba(124,58,237,0.35);background:rgba(124,58,237,0.06);transform:translateX(4px)}
.opt.sel{border-color:rgba(124,58,237,0.45);background:rgba(124,58,237,0.08)}
.opt.corr{border-color:rgba(16,185,129,0.45);background:rgba(16,185,129,0.07)}
.opt.wrong{border-color:rgba(248,113,113,0.35);background:rgba(248,113,113,0.06)}
.opt:disabled{cursor:default}
.opt-key{width:26px;height:26px;border-radius:8px;flex-shrink:0;border:1px solid var(--rim);background:var(--ink3);font-family:'DM Mono',monospace;font-size:10px;font-weight:500;display:flex;align-items:center;justify-content:center;transition:all 0.15s;margin-top:1px}
.opt.sel .opt-key{border-color:var(--violet);background:var(--violet);color:#fff;box-shadow:0 0 12px rgba(124,58,237,0.4)}
.opt.corr .opt-key{border-color:var(--emerald);background:var(--emerald);color:#fff;box-shadow:0 0 12px rgba(16,185,129,0.4)}
.opt.wrong .opt-key{border-color:#F87171;background:#F87171;color:#fff}
.short-ta{width:100%;min-height:120px;padding:16px 18px;border:1px solid var(--rim);border-radius:14px;background:var(--ink2);color:var(--text);resize:vertical;font-family:'Syne',sans-serif;font-size:14px;line-height:1.7;outline:none;transition:all 0.2s}
.short-ta:focus{border-color:rgba(124,58,237,0.4);box-shadow:0 0 0 3px rgba(124,58,237,0.08);background:var(--ink3)}
.short-ta:disabled{opacity:0.7}
.hint-row{display:flex;align-items:flex-start;gap:8px;margin-top:10px;padding:10px 14px;border-radius:10px;background:rgba(245,158,11,0.05);border:1px solid rgba(245,158,11,0.12)}
.hint-ico{font-size:14px;flex-shrink:0;margin-top:1px}
.hint-txt{font-family:'DM Mono',monospace;font-size:11px;color:rgba(245,158,11,0.8);line-height:1.6}
.conf-row{display:flex;align-items:center;gap:8px;margin-top:16px;flex-wrap:wrap}
.conf-label{font-family:'DM Mono',monospace;font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:0.12em;white-space:nowrap}
.conf-btn{padding:5px 13px;border-radius:20px;border:1px solid var(--rim);background:transparent;font-family:'Syne',sans-serif;font-size:11px;font-weight:700;cursor:pointer;transition:all 0.18s;color:var(--text3)}
.conf-btn:hover{border-color:var(--rim2);color:var(--text)}
.conf-btn.high{background:rgba(16,185,129,0.12);border-color:rgba(16,185,129,0.25);color:var(--emerald)}
.conf-btn.high.on{background:var(--emerald);color:#fff;border-color:var(--emerald);box-shadow:0 0 12px rgba(16,185,129,0.4)}
.conf-btn.med{background:rgba(245,158,11,0.1);border-color:rgba(245,158,11,0.22);color:var(--gold)}
.conf-btn.med.on{background:var(--gold);color:#fff;border-color:var(--gold)}
.conf-btn.low{background:rgba(248,113,113,0.1);border-color:rgba(248,113,113,0.22);color:#F87171}
.conf-btn.low.on{background:#F87171;color:#fff;border-color:#F87171}

/* SUBMIT BAR */
.sub-bar{position:fixed;bottom:0;left:0;right:0;z-index:100;background:rgba(3,2,10,0.88);backdrop-filter:blur(32px) saturate(1.5);border-top:1px solid var(--rim);padding:14px 40px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.sb-progress{flex:1;display:flex;flex-direction:column;gap:7px;min-width:120px}
.sb-bar{height:3px;background:var(--ink3);border-radius:2px;overflow:hidden}
.sb-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--violet),var(--cyan));transition:width 0.4s cubic-bezier(0.16,1,0.3,1);box-shadow:0 0 8px var(--violet)}
.sb-meta{font-family:'DM Mono',monospace;font-size:10px;color:var(--text3)}
.sb-meta strong{color:var(--violet2)}
.sb-flags{font-family:'DM Mono',monospace;font-size:10px;color:var(--gold);display:flex;align-items:center;gap:6px}
.sb-actions{display:flex;gap:8px}

/* EVAL OVERLAY */
.eval-overlay{position:fixed;inset:0;z-index:300;background:rgba(3,2,10,0.96);backdrop-filter:blur(24px);display:flex;flex-direction:column;align-items:center;justify-content:center;animation:fade-in 0.4s ease both}
.eval-ring{width:80px;height:80px;margin:0 auto 32px;position:relative}
.er-o{position:absolute;inset:0;border:3px solid transparent;border-top-color:var(--violet);border-right-color:rgba(124,58,237,0.4);border-radius:50%;animation:spin 0.9s linear infinite;box-shadow:0 0 40px rgba(124,58,237,0.3)}
.er-m{position:absolute;inset:12px;border:2px solid transparent;border-bottom-color:var(--cyan);border-left-color:rgba(6,182,212,0.4);border-radius:50%;animation:spin-rev 0.6s linear infinite}
.er-i{position:absolute;inset:24px;border:1px solid transparent;border-top-color:var(--rose);border-radius:50%;animation:spin 1.5s linear infinite}
.eval-title{font-size:28px;font-weight:800;letter-spacing:-0.8px;margin-bottom:8px}
.eval-sub{font-size:14px;color:var(--text2);margin-bottom:36px}
.eval-prog{width:380px;max-width:90vw}
.eval-bar-wrap{height:5px;background:var(--ink3);border-radius:3px;overflow:hidden;margin-bottom:12px}
.eval-bar{height:100%;background:linear-gradient(90deg,var(--violet),var(--cyan));border-radius:3px;transition:width 0.6s cubic-bezier(0.16,1,0.3,1);box-shadow:0 0 12px rgba(124,58,237,0.5)}
.eval-step{font-family:'DM Mono',monospace;font-size:11px;color:var(--text3);text-align:center}

/* RESULTS */
.results-header{text-align:center;padding:64px 0 52px;position:relative}
.rank-shell{display:flex;flex-direction:column;align-items:center;justify-content:center;width:156px;height:156px;border-radius:50%;margin:0 auto 28px;position:relative;border:2px solid currentColor;animation:pop 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.2s both}
.rank-shell::before{content:'';position:absolute;inset:-6px;border-radius:50%;background:conic-gradient(from 0deg,transparent 0%,currentColor 20%,transparent 40%);opacity:0.2;animation:halo-spin 6s linear infinite}
.rank-shell::after{content:'';position:absolute;inset:-14px;border-radius:50%;border:1px dashed currentColor;opacity:0.1;animation:halo-spin 12s linear infinite reverse}
.rank-glow{position:absolute;inset:-30px;border-radius:50%;background:radial-gradient(circle,var(--glow) 0%,transparent 70%);opacity:0.5}
.rank-letter{font-size:54px;font-weight:900;line-height:1;letter-spacing:-3px;position:relative;z-index:1}
.rank-pct{font-family:'DM Mono',monospace;font-size:12px;opacity:0.65;margin-top:2px;position:relative;z-index:1}
.res-emoji{font-size:34px;margin-bottom:12px;display:block;animation:float 3s ease infinite}
.res-name{font-family:'DM Mono',monospace;font-size:12px;color:var(--text3);margin-bottom:8px;letter-spacing:0.06em}
.res-rank-label{font-size:42px;font-weight:900;letter-spacing:-2px;margin-bottom:10px}
.res-pts{font-size:15px;color:var(--text2)}

.stat-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--rim);border:1px solid var(--rim);border-radius:18px;overflow:hidden;margin:36px 0}
.stat{background:var(--ink1);padding:22px 12px;text-align:center;transition:background 0.2s}
.stat:hover{background:var(--ink2)}
.stat-n{font-size:30px;font-weight:800;letter-spacing:-1px;line-height:1}
.stat-n.g{color:var(--emerald);text-shadow:0 0 20px rgba(16,185,129,0.3)}
.stat-n.y{color:var(--gold);text-shadow:0 0 20px rgba(245,158,11,0.3)}
.stat-n.r{color:#F87171;text-shadow:0 0 20px rgba(248,113,113,0.3)}
.stat-n.p{color:var(--violet2);text-shadow:0 0 20px rgba(167,139,250,0.3)}
.stat-n.b{color:var(--sky);text-shadow:0 0 20px rgba(56,189,248,0.3)}
.stat-l{font-family:'DM Mono',monospace;font-size:8px;color:var(--text3);text-transform:uppercase;letter-spacing:0.1em;margin-top:6px}

.topic-breakdown{margin-bottom:36px}
.section-hd{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:0.16em;color:var(--text3);margin-bottom:18px;display:flex;align-items:center;gap:12px}
.section-hd::after{content:'';flex:1;height:1px;background:var(--rim)}
.tb-row{display:flex;align-items:center;gap:12px;margin-bottom:10px}
.tb-name{font-size:13px;font-weight:600;min-width:140px;color:var(--text2)}
.tb-wrap{flex:1;height:7px;background:var(--ink3);border-radius:4px;overflow:hidden}
.tb-fill{height:100%;border-radius:4px;animation:bar-fill 1s cubic-bezier(0.16,1,0.3,1)}
.tb-pct{font-family:'DM Mono',monospace;font-size:10px;color:var(--text3);min-width:36px;text-align:right}

/* CHARTS SECTION */
.charts-section{margin-bottom:36px}
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:0}
.chart-card{background:var(--ink1);border:1px solid var(--rim);border-radius:18px;padding:22px;overflow:hidden}
.chart-card.full{grid-column:1/-1}
.chart-title{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:0.14em;color:var(--text3);margin-bottom:18px;display:flex;align-items:center;gap:8px}
.chart-title::before{content:'';width:3px;height:12px;border-radius:2px;background:var(--violet)}

/* AI SUMMARY */
.ai-card{padding:30px 34px;border-radius:20px;border:1px solid rgba(124,58,237,0.18);background:linear-gradient(135deg,rgba(124,58,237,0.04),rgba(6,182,212,0.02));margin-bottom:36px;position:relative;overflow:hidden}
.ai-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(124,58,237,0.4),rgba(6,182,212,0.4),transparent)}
.ai-lbl{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:0.16em;color:var(--violet2);margin-bottom:14px;display:flex;align-items:center;gap:10px}
.ai-lbl::before{content:'';width:24px;height:1px;background:var(--violet2)}
.ai-txt{font-size:15px;line-height:1.85;color:var(--text2)}

/* RECOMMENDATIONS */
.reco-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:36px}
.reco-card{padding:18px 20px;border-radius:16px;border:1px solid var(--rim);background:var(--ink2);transition:all 0.2s}
.reco-card:hover{border-color:var(--rim2);background:var(--ink3);transform:translateY(-2px)}
.reco-type{font-family:'DM Mono',monospace;font-size:8px;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px;display:flex;align-items:center;gap:6px}
.reco-topic{font-size:14px;font-weight:700;margin-bottom:4px;letter-spacing:-0.2px}
.reco-action{font-size:12px;color:var(--text3);line-height:1.6}
.reco-priority{font-family:'DM Mono',monospace;font-size:8px;padding:2px 8px;border-radius:4px;letter-spacing:0.06em}

.export-row{display:flex;align-items:center;gap:10px;margin-bottom:28px;padding:14px 18px;border-radius:14px;background:var(--glass);border:1px solid var(--rim);flex-wrap:wrap}
.er-label{font-family:'DM Mono',monospace;font-size:10px;color:var(--text3);flex:1}

/* REVIEW CARDS */
.rev-card{margin-bottom:12px;padding:22px 22px;border-radius:18px;border:1px solid var(--rim);background:var(--ink1);animation:slide-up 0.4s ease both;position:relative;overflow:hidden}
.rev-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px}
.rev-card.correct::before{background:var(--emerald);box-shadow:0 0 12px rgba(16,185,129,0.5)}
.rev-card.incorrect::before{background:#F87171;box-shadow:0 0 12px rgba(248,113,113,0.4)}
.rev-card.partial::before{background:var(--gold);box-shadow:0 0 12px rgba(245,158,11,0.4)}
.rev-card.correct{border-color:rgba(16,185,129,0.18)}
.rev-card.incorrect{border-color:rgba(248,113,113,0.15)}
.rev-card.partial{border-color:rgba(245,158,11,0.18)}
.rev-pills{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:12px}
.rpill{font-family:'DM Mono',monospace;font-size:9px;padding:3px 10px;border-radius:20px;text-transform:uppercase;letter-spacing:0.07em}
.rpill.c{background:rgba(16,185,129,0.1);color:var(--emerald)}
.rpill.i{background:rgba(248,113,113,0.1);color:#F87171}
.rpill.p{background:rgba(245,158,11,0.1);color:var(--gold)}
.rpill.s{background:rgba(124,58,237,0.1);color:var(--violet2)}
.rpill.f{background:rgba(245,158,11,0.07);color:var(--gold);border:1px solid rgba(245,158,11,0.18)}
.rev-q{font-size:14px;font-weight:700;margin-bottom:8px;line-height:1.45}
.rev-ans{font-size:13px;color:var(--text3);margin-bottom:4px}
.rev-ans em{color:var(--text2);font-style:italic}
.rev-fb{font-size:13px;color:var(--text3);line-height:1.7;margin-top:12px;padding-top:12px;border-top:1px solid var(--rim)}

/* LEADERBOARD */
.lb-card{padding:24px;border-radius:20px;border:1px solid var(--rim);background:var(--ink1);margin-top:32px}
.lb-hd{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:0.16em;color:var(--text3);margin-bottom:18px}
.lb-row{display:flex;align-items:center;gap:12px;padding:11px 14px;border-radius:12px;margin-bottom:6px;background:var(--glass);transition:background 0.2s}
.lb-row:hover{background:var(--glass2)}
.lb-row.me{background:rgba(124,58,237,0.07);border:1px solid rgba(124,58,237,0.18)}
.lb-pos{font-family:'DM Mono',monospace;font-size:13px;font-weight:700;width:26px;color:var(--text3)}
.lb-pos.g{color:var(--gold)} .lb-pos.s{color:#C0C0C0} .lb-pos.b{color:#CD7F32}
.lb-name{flex:1;font-weight:700;font-size:14px}
.lb-score{font-family:'DM Mono',monospace;font-size:11px;color:var(--violet2)}
.lb-badge{font-size:12px;font-weight:700;padding:2px 9px;border-radius:7px}
.res-acts{display:flex;gap:10px;margin-top:32px;flex-wrap:wrap}

/* TOAST */
.toast{position:fixed;bottom:90px;right:24px;z-index:400;padding:13px 20px;border-radius:14px;background:var(--ink2);border:1px solid var(--rim2);color:var(--text);font-family:'DM Mono',monospace;font-size:12px;box-shadow:0 24px 64px rgba(0,0,0,0.6);animation:slide-up 0.3s cubic-bezier(0.16,1,0.3,1) both;max-width:320px;display:flex;align-items:center;gap:10px;backdrop-filter:blur(20px)}

@media(max-width:640px){
  .nav{padding:12px 16px} .nav-steps{display:none}
  .wrap{padding:36px 16px 130px}
  .stat-grid{grid-template-columns:repeat(3,1fr)} .cfg{grid-template-columns:1fr 1fr}
  .doc-stats{grid-template-columns:repeat(2,1fr)} .feat-grid{grid-template-columns:repeat(2,1fr)}
  .quiz-header{flex-direction:column} .sub-bar{padding:12px 16px}
  .hero-h{letter-spacing:-2px} .rag-grid{grid-template-columns:repeat(2,1fr)}
  .chart-grid{grid-template-columns:1fr} .reco-grid{grid-template-columns:1fr}
}
`;

// ─── TECH STACK CHIPS ─────────────────────────────────────────────────────
const TECH_CHIPS = [
  { label: "RAG",         color: "#818CF8", bg: "rgba(129,140,248,0.12)" },
  { label: "Chunking",    color: "#34D399", bg: "rgba(52,211,153,0.12)"  },
  { label: "Embeddings",  color: "#E879F9", bg: "rgba(232,121,249,0.12)" },
  { label: "VectorDB",    color: "#38BDF8", bg: "rgba(56,189,248,0.12)"  },
  { label: "OCR",         color: "#FBBF24", bg: "rgba(251,191,36,0.12)"  },
  { label: "PDF Vision",  color: "#F87171", bg: "rgba(248,113,113,0.12)" },
  { label: "Vision MLLM", color: "#FB923C", bg: "rgba(251,146,60,0.12)"  },
  { label: "Gemini API",  color: "#60A5FA", bg: "rgba(96,165,250,0.12)"  },
  { label: "MongoDB",     color: "#4ADE80", bg: "rgba(74,222,128,0.12)"  },
  { label: "Recharts",    color: "#F472B6", bg: "rgba(244,114,182,0.12)" },
  { label: "AI Reco",     color: "#A78BFA", bg: "rgba(167,139,250,0.12)" },
  { label: "Docker",      color: "#67E8F9", bg: "rgba(103,232,249,0.12)" },
];

const GEN_STEPS = [
  "Chunking document (RAG)…",
  "Building TF-IDF embeddings…",
  "Indexing vector store (MongoDB)…",
  "Retrieving top-k chunks…",
  "Prompting Gemini API…",
  "Generating questions…",
  "Finalising quiz…",
];

// ─────────────────────────────────────────────────────────────────────────
//  MAIN APP
// ─────────────────────────────────────────────────────────────────────────
export default function App() {
  const [stage, setStage]         = useState("name");
  const [userName, setUser]       = useState("");
  const [docText, setDocText]     = useState("");
  const [paste, setPaste]         = useState("");
  const [fileName, setFile]       = useState("");
  const [numQ, setNumQ]           = useState("5");
  const [qType, setQType]         = useState("mixed");
  const [difficulty, setDiff]     = useState("mixed");
  const [timerOn, setTimerOn]     = useState(false);
  const [timeLimit, setTimeL]     = useState("300");
  const [drag, setDrag]           = useState(false);
  const [genStep, setGenStep]     = useState(0);
  const [evalPct, setEvalPct]     = useState(0);
  const [evalTxt, setEvalTxt]     = useState("");
  const [questions, setQ]         = useState([]);
  const [topics, setTopics]       = useState([]);
  const [answers, setAns]         = useState({});
  const [confidence, setConf]     = useState({});
  const [flagged, setFlagged]     = useState({});
  const [feedbacks, setFb]        = useState({});
  const [summary, setSummary]     = useState("");
  const [timer, setTimer]         = useState(0);
  const [timerActive, setTA]      = useState(false);
  const [toast, setToast]         = useState("");
  const [lb, setLB]               = useState(LEADERBOARD);
  const [ragInfo, setRagInfo]     = useState(null);
  const [recos, setRecos]         = useState([]);
  const [ocrActive, setOcrActive] = useState(false);
  const [ocrType, setOcrType]     = useState("image"); // "image" | "pdf"

  const fileRef  = useRef();
  const timerRef = useRef();

  const msg = (m, icon = "ℹ") => { setToast({ m, icon }); setTimeout(() => setToast(""), 3800); };

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

  // ── File upload — supports text, images (OCR), and PDFs (Gemini Vision) ──
  const handleFile = async f => {
    if (!f) return;

    const isImage = f.type.startsWith("image/");
    const isPDF   = f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf");

    if (isPDF) {
      // ── PDF: send as base64 to Gemini Vision which natively understands PDFs ──
      msg("PDF detected — extracting text via Gemini Vision…", "📄");
      setOcrActive(true);
      setOcrType("pdf");
      try {
        const b64 = await readFileAsBase64(f);
        const extracted = await ocrFile(b64, "application/pdf");
        if (!extracted || extracted.trim().length < 20) {
          throw new Error("No text could be extracted from this PDF.");
        }
        setDocText(extracted);
        setFile(f.name + " [PDF→text]");
        msg(`PDF extracted — ${extracted.split(/\s+/).filter(Boolean).length.toLocaleString()} words found`, "✅");
      } catch (e) {
        msg("PDF extraction failed: " + e.message, "❌");
      }
      setOcrActive(false);
      setOcrType("image");

    } else if (isImage) {
      // ── Image: OCR via Gemini Vision ──
      msg("Image detected — running OCR + Vision…", "👁");
      setOcrActive(true);
      setOcrType("image");
      try {
        const b64 = await readFileAsBase64(f);
        const extracted = await ocrFile(b64, f.type);
        setDocText(extracted);
        setFile(f.name + " [OCR]");
        msg("OCR complete — text extracted!", "✅");
      } catch (e) {
        msg("OCR failed: " + e.message, "❌");
      }
      setOcrActive(false);

    } else {
      // ── Plain text file ──
      try {
        const t = await readFileAsText(f);
        setDocText(t);
        setFile(f.name);
      } catch { msg("Could not read file", "❌"); }
    }
  };

  const activeText = docText || paste;
  const wordCount  = activeText ? activeText.split(/\s+/).filter(Boolean).length : 0;
  const sentCount  = activeText ? activeText.split(/[.!?]+/).filter(Boolean).length : 0;
  const paraCount  = activeText ? activeText.split(/\n\n+/).filter(Boolean).length : 0;
  const charCount  = activeText.length;

  // ── RAG: index whenever text changes ──
  useEffect(() => {
    if (activeText.length > 100) {
      const t = setTimeout(() => {
        const chunks = chunkText(activeText);
        const vocab  = buildVocab(chunks);
        const idf    = buildIdf(chunks, vocab);
        vectorStore.insertMany(chunks, vocab, idf);
        setRagInfo({ chunks: chunks.length, vocab: Object.keys(vocab).length, idf });
        if (topics.length === 0) extractTopics();
        console.log("[MongoDB]", vectorStore.dbInfo);
      }, 800);
      return () => clearTimeout(t);
    }
  }, [activeText]);

  const extractTopics = async () => {
    const text = activeText.trim();
    if (text.length < 80) return;
    try {
      const raw = await gemini(
        `Extract 5-8 key topics from this document. Return ONLY a JSON array of short strings (2-4 words each).\n\nDocument:\n${text.slice(0, 3000)}`,
        "You are a document analyst. Return only valid JSON arrays.", 200
      );
      const arr = parseJSON(raw);
      if (Array.isArray(arr)) setTopics(arr);
    } catch {}
  };

  // ── Generate quiz using RAG ──
  const generate = async () => {
    const text = activeText.trim();
    if (!text || text.length < 50) { msg("Please provide at least 50 characters", "⚠"); return; }
    setStage("gen-loading"); setGenStep(0);
    const iv = setInterval(() => setGenStep(s => Math.min(s + 1, GEN_STEPS.length - 1)), 900);

    const topicQuery = topics.slice(0, 3).join(" ") || text.slice(0, 200);
    const retrieved  = vectorStore.vectorSearch(topicQuery, 5);
    const ragContext = retrieved.map(d => d.text).join("\n\n---\n\n");
    const context    = ragContext.length > 200 ? ragContext : text.slice(0, 9000);

    const sys = `You are an expert quiz generator. Return ONLY a valid JSON array.
Schema per item:
{ "id":number, "type":"mcq"|"short", "topic":string, "difficulty":"easy"|"medium"|"hard",
  "question":string,
  MCQ: "options":[4 strings], "answer":string (exact match), "explanation":string,
  Short: "answer":string (model answer 1-3 sentences), "keywords":[strings], "hint":string }
Rules:
- ALL questions MUST come strictly from the provided context.
- Generate exactly ${numQ} questions.
- Type: ${qType === "mcq" ? "ALL mcq" : qType === "short" ? "ALL short" : "mix mcq and short"}.
- Difficulty: ${difficulty === "mixed" ? "mix easy,medium,hard" : difficulty}.
- Make distractors plausible and educational.`;

    try {
      const raw = await gemini(
        `Context (RAG-retrieved passages):\n\n${context}\n\nGenerate ${numQ} quiz questions now.`,
        sys, 3000
      );
      clearInterval(iv);
      const qs = parseJSON(raw);
      if (!Array.isArray(qs)) { msg("Could not parse questions. Try again.", "❌"); setStage("upload"); return; }
      setQ(qs); setAns({}); setFb({}); setConf({}); setFlagged({}); setSummary(""); setRecos([]);
      if (timerOn) { setTimer(parseInt(timeLimit)); setTA(true); }
      setStage("quiz");
    } catch { clearInterval(iv); msg("API error. Please try again.", "❌"); setStage("upload"); }
  };

  const answeredCount = Object.keys(answers).filter(k => answers[k] !== undefined && answers[k] !== "").length;
  const flaggedCount  = Object.values(flagged).filter(Boolean).length;

  // ── Evaluate ──
  const evaluateAll = async () => {
    clearTimeout(timerRef.current); setTA(false);
    setStage("eval-loading"); setEvalPct(0);
    const fb = {};
    const mcqQs   = questions.filter(q => q.type === "mcq");
    const shortQs = questions.filter(q => q.type === "short");
    const total   = 5 + shortQs.length;
    let done = 0;
    const tick = txt => { done++; setEvalPct(Math.round((done / total) * 100)); setEvalTxt(txt); };

    tick("Parsing responses…");
    await new Promise(r => setTimeout(r, 300));
    tick("Evaluating MCQs…");
    mcqQs.forEach(q => {
      const idx = questions.indexOf(q);
      const ua  = answers[idx] || "";
      const ok  = ua === q.answer;
      fb[idx] = { status: ok ? "correct" : "incorrect", text: ok ? (q.explanation || "Correct!") : `Correct answer: ${q.answer}. ${q.explanation || ""}`.trim(), score: ok ? 10 : 0, modelAns: q.answer };
    });
    await new Promise(r => setTimeout(r, 200));

    for (const q of shortQs) {
      const idx = questions.indexOf(q);
      const ua  = (answers[idx] || "").trim();
      tick(`Grading via Gemini: "${q.question.slice(0, 35)}…"`);
      if (!ua) { fb[idx] = { status: "incorrect", text: "No answer provided.", score: 0, modelAns: q.answer }; continue; }
      try {
        const raw = await gemini(
          `Q: ${q.question}\nModel: ${q.answer}\nKeywords: ${(q.keywords || []).join(", ")}\nStudent: ${ua}\n\nReturn ONLY JSON: {"status":"correct"|"partial"|"incorrect","score":0-10,"feedback":"2-3 sentences","modelAnswer":"${q.answer}"}`,
          "You are a rigorous academic evaluator. Return ONLY valid JSON.", 350
        );
        const p = parseJSON(raw);
        fb[idx] = { status: p.status, text: p.feedback, score: p.score, modelAns: q.answer };
      } catch { fb[idx] = { status: "partial", text: "Evaluation error.", score: 5, modelAns: q.answer }; }
    }
    setFb(fb);

    tick("Computing breakdown…");
    await new Promise(r => setTimeout(r, 200));
    const totPts = Object.values(fb).reduce((s, f) => s + (f?.score || 0), 0);
    const maxPts = questions.length * 10;
    const pct    = Math.round((totPts / maxPts) * 100);

    const tbMap = {};
    questions.forEach((q, i) => {
      if (!tbMap[q.topic]) tbMap[q.topic] = { total: 0, got: 0 };
      tbMap[q.topic].total += 10;
      tbMap[q.topic].got   += fb[i]?.score || 0;
    });
    const tbArr = Object.entries(tbMap).map(([t, { total, got }]) => ({ topic: t, pct: Math.round((got / total) * 100), got, total }));

    tick("Generating AI summary…");
    const info     = questions.map((q, i) => `Q${i + 1}[${q.topic}/${q.difficulty}]:${fb[i]?.status || "?"}`).join(" | ");
    const confInfo = Object.entries(confidence).map(([i, c]) => `Q${+i + 1}:${c}`).join(", ");
    try {
      const t = await gemini(
        `Student: "${userName}". Score: ${pct}% (${totPts}/${maxPts}). Rank: ${getRank(pct).rank}.\nPer-question: ${info}.\nConfidence: ${confInfo || "not provided"}.\nWrite 4-5 sentences of honest, personalised feedback. Address them by name. Highlight strengths, weaknesses, and next steps.`,
        "You are a supportive academic tutor. Be specific, warm, actionable. Plain text.", 400
      );
      setSummary(t);
    } catch { setSummary(`${userName}, you scored ${pct}%. Review the questions below to deepen your understanding.`); }

    tick("Generating AI recommendations…");
    try { const r = await generateRecommendations(tbArr, userName, pct); setRecos(r); } catch {}

    MongoDB.save({
      _id: `session_${Date.now()}`,
      user: userName, score: pct, pts: totPts, max: maxPts,
      rank: getRank(pct).rank, topics: tbArr,
      ragChunks: ragInfo?.chunks || 0,
      timestamp: new Date().toISOString(),
    });

    const rank = getRank(pct);
    saveLB({ name: userName, pct, pts: totPts, max: maxPts, rank: rank.rank, color: rank.color, time: new Date().toLocaleTimeString() });
    setLB([...LEADERBOARD]);
    setStage("results");
  };

  // ── Computed results ──
  const topicBreakdown = () => {
    const map = {};
    questions.forEach((q, i) => {
      if (!map[q.topic]) map[q.topic] = { total: 0, got: 0 };
      map[q.topic].total += 10;
      map[q.topic].got   += feedbacks[i]?.score || 0;
    });
    return Object.entries(map).map(([t, { total, got }]) => ({ topic: t, pct: Math.round((got / total) * 100), got, total }));
  };

  const totalPts = Object.values(feedbacks).reduce((s, f) => s + (f?.score || 0), 0);
  const maxPts   = questions.length * 10;
  const pct      = maxPts > 0 ? Math.round((totalPts / maxPts) * 100) : 0;
  const nCorr    = Object.values(feedbacks).filter(f => f?.status === "correct").length;
  const nPart    = Object.values(feedbacks).filter(f => f?.status === "partial").length;
  const nWrong   = Object.values(feedbacks).filter(f => f?.status === "incorrect").length;
  const rank     = getRank(pct);
  const avgTime  = timerOn && parseInt(timeLimit) > timer ? Math.round((parseInt(timeLimit) - timer) / questions.length) : null;

  // ── Chart data ──
  const radarData = useMemo(() => topicBreakdown().map(t => ({ topic: t.topic.slice(0, 12), score: t.pct, full: 100 })), [feedbacks]);
  const barData   = useMemo(() => [
    { name: "Correct", value: nCorr, fill: "#34D399" },
    { name: "Partial", value: nPart, fill: "#FBBF24" },
    { name: "Wrong",   value: nWrong, fill: "#F87171" },
  ], [feedbacks]);
  const diffData = useMemo(() => {
    const map = { easy: { correct: 0, total: 0 }, medium: { correct: 0, total: 0 }, hard: { correct: 0, total: 0 } };
    questions.forEach((q, i) => {
      const d = q.difficulty || "medium";
      map[d].total++;
      if (feedbacks[i]?.status === "correct") map[d].correct++;
    });
    return Object.entries(map).map(([d, v]) => ({ diff: d, pct: v.total ? Math.round((v.correct / v.total) * 100) : 0, total: v.total }));
  }, [feedbacks]);

  const exportResults = () => {
    const tb = topicBreakdown();
    const lines = [
      `QUIZ RESULTS — ${userName}`, `Date: ${new Date().toLocaleString()}`,
      `Score: ${totalPts}/${maxPts} (${pct}%) — Rank: ${rank.rank} (${rank.label})`,
      `Correct: ${nCorr} | Partial: ${nPart} | Incorrect: ${nWrong}`, "",
      `RAG: ${ragInfo?.chunks || 0} chunks, ${ragInfo?.vocab || 0} vocab terms`,
      `MongoDB: ${MongoDB.sessions.length} sessions stored`, "",
      "=== TOPIC BREAKDOWN ===",
      ...tb.map(t => `${t.topic}: ${t.pct}% (${t.got}/${t.total})`), "",
      "=== RECOMMENDATIONS ===",
      ...recos.map(r => `[${r.type.toUpperCase()}] ${r.topic}: ${r.action}`), "",
      "=== QUESTION REVIEW ===",
      ...questions.map((q, i) => {
        const fb = feedbacks[i];
        return [`Q${i + 1} [${q.topic}/${q.difficulty}] — ${fb?.status || "?"} (${fb?.score || 0}/10)`,
          `Q: ${q.question}`, `Your Answer: ${answers[i] || "(none)"}`,
          q.type === "mcq" ? `Correct: ${q.answer}` : "",
          `Feedback: ${fb?.text || ""}`, ""].filter(Boolean).join("\n");
      }), "", "=== AI SUMMARY ===", summary,
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = `quiz_${userName}_${Date.now()}.txt`; a.click();
    msg("Results exported!", "✅");
  };

  const restart = () => {
    setStage("name"); setUser(""); setDocText(""); setPaste(""); setFile("");
    setQ([]); setAns({}); setFb({}); setConf({}); setFlagged({}); setSummary("");
    setTopics([]); setTA(false); setTimer(0); setRagInfo(null); setRecos([]);
  };
  const retake = () => {
    setAns({}); setFb({}); setConf({}); setFlagged({}); setSummary(""); setRecos([]);
    if (timerOn) { setTimer(parseInt(timeLimit)); setTA(true); }
    setStage("quiz");
  };

  const stageMap    = { name: 0, upload: 1, "gen-loading": 2, quiz: 2, "eval-loading": 3, results: 3 };
  const si          = stageMap[stage] ?? 0;
  const stageLabels = ["Profile", "Document", "Quiz", "Results"];

  const recoStyle = type => {
    const m = { review: { color: "#F87171", bg: "rgba(248,113,113,0.08)" }, practice: { color: "#FBBF24", bg: "rgba(251,191,36,0.08)" }, advance: { color: "#34D399", bg: "rgba(52,211,153,0.08)" }, resource: { color: "#818CF8", bg: "rgba(129,140,248,0.08)" } };
    return m[type] || m.review;
  };
  const prioStyle = p => ({ high: { color: "#F87171", bg: "rgba(248,113,113,0.12)" }, medium: { color: "#FBBF24", bg: "rgba(251,191,36,0.1)" }, low: { color: "#34D399", bg: "rgba(52,211,153,0.1)" } }[p] || {});

  return (
    <>
      <style>{CSS}</style>

      <div className="bg-canvas">
        <div className="grid-bg"/>
        <div className="stars"/>
        <div className="nebula nb1"/><div className="nebula nb2"/>
        <div className="nebula nb3"/><div className="nebula nb4"/>
        <div className="scanlines"/>
      </div>

      <div className="page">

        {/* Eval overlay */}
        {stage === "eval-loading" && (
          <div className="eval-overlay">
            <div className="eval-ring"><div className="er-o"/><div className="er-m"/><div className="er-i"/></div>
            <div className="eval-title">Evaluating your answers</div>
            <div className="eval-sub">RAG · Vector Search · Gemini grading · MongoDB logging</div>
            <div className="eval-prog">
              <div className="eval-bar-wrap"><div className="eval-bar" style={{ width: `${evalPct}%` }}/></div>
              <div className="eval-step">{evalTxt}</div>
            </div>
          </div>
        )}

        {/* NAV */}
        <nav className="nav">
          <div className="nav-logo">
            <div className="nav-mark">⚡</div>
            <div className="nav-wordmark">Quiz<em>.AI</em></div>
          </div>
          <div className="nav-tag">Gemini 2.5 Flash</div>
          <div className="nav-r">
            {userName && (
              <div className="nav-user-chip">
                <div className="nav-user-avatar">👤</div>
                {userName}
              </div>
            )}
            <div className="nav-steps">
              {stageLabels.map((l, i) => (
                <div key={l} className={`nstep ${i < si ? "done" : i === si ? "on" : ""}`}>{l}</div>
              ))}
            </div>
          </div>
        </nav>

        {/* TECH STACK BANNER */}
        <div className="tech-banner">
          <span className="tb-label">Stack</span>
          <div className="tb-sep"/>
          {TECH_CHIPS.map(t => (
            <span key={t.label} className="tech-chip"
              style={{ color: t.color, borderColor: t.color + "40", background: t.bg }}>
              {t.label}
            </span>
          ))}
        </div>

        <div className="wrap">

          {/* ── NAME ── */}
          {stage === "name" && (
            <div className="hero">
              <div className="hero-eyebrow">
                <span className="hero-eyebrow-dot"/>
                RAG · Embeddings · Vision OCR · PDF · MongoDB · Recharts
              </div>
              <h1 className="hero-h">
                <span className="l1">Learn smarter with</span>
                <span className="l2">intelligent quizzing</span>
              </h1>
              <p className="hero-p">
                Upload any document, image, or PDF. The RAG pipeline chunks it, builds TF-IDF embeddings, indexes them in a vector store, retrieves the most relevant passages, then calls Gemini to generate and grade questions — all results saved to MongoDB.
              </p>
              <div className="name-card">
                <label className="nc-label" htmlFor="uname">Your name to get started</label>
                <input id="uname" className="nc-input" placeholder="e.g. Arjun Sharma"
                  value={userName} onChange={e => setUser(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && userName.trim() && setStage("upload")} autoFocus/>
                <button className="btn btn-prime" style={{ width: "100%", justifyContent: "center", fontSize: 15 }}
                  onClick={() => setStage("upload")} disabled={!userName.trim()}>
                  Begin your journey →
                </button>
              </div>
              <div className="feat-grid">
                {[
                  { ico: "🔪", txt: "Text Chunking" }, { ico: "🔢", txt: "Embeddings" },
                  { ico: "🔍", txt: "Vector Search" }, { ico: "📄", txt: "PDF Support"  },
                  { ico: "👁", txt: "OCR + Vision" },  { ico: "🧠", txt: "Gemini LLM"  },
                  { ico: "📊", txt: "Recharts viz" },  { ico: "🤖", txt: "AI Reco"      },
                ].map(f => (
                  <div key={f.txt} className="feat-item">
                    <span className="feat-ico">{f.ico}</span>
                    <span className="feat-txt">{f.txt}</span>
                  </div>
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
                <p className="up-sub">Text files, PDFs, images (OCR), or paste directly. The RAG pipeline will chunk, embed and index automatically.</p>
              </div>

              {/* RAG Status Panel */}
              {ragInfo && (
                <div className="rag-panel">
                  <div className="rag-title"><span className="rdot"/>RAG Pipeline — Active</div>
                  <div className="rag-grid">
                    <div className="rag-stat"><div className="rag-n">{ragInfo.chunks}</div><div className="rag-l">Chunks</div></div>
                    <div className="rag-stat"><div className="rag-n">{ragInfo.vocab.toLocaleString()}</div><div className="rag-l">Vocab terms</div></div>
                    <div className="rag-stat"><div className="rag-n">{vectorStore.size}</div><div className="rag-l">Vectors stored</div></div>
                    <div className="rag-stat"><div className="rag-n">{MongoDB.sessions.length}</div><div className="rag-l">MongoDB docs</div></div>
                  </div>
                </div>
              )}

              {fileName && (
                <div className="file-pill">
                  <div className="fp-icon">{fileName.includes("[PDF") ? "📄" : fileName.includes("[OCR]") ? "🖼️" : "📝"}</div>
                  <div className="fp-info">
                    <div className="fp-name">{fileName}</div>
                    <div className="fp-meta">{(charCount / 1024).toFixed(1)} KB · {wordCount.toLocaleString()} words · {sentCount} sentences</div>
                  </div>
                  <button className="fp-rm" onClick={() => { setDocText(""); setFile(""); setTopics([]); setRagInfo(null); }}>✕</button>
                </div>
              )}

              {/* OCR / PDF processing notice */}
              {ocrActive && (
                <div className={`ocr-notice ${ocrType === "pdf" ? "pdf-notice" : ""}`}>
                  <span>{ocrType === "pdf" ? "📄" : "👁"}</span>
                  {ocrType === "pdf"
                    ? "Extracting text from PDF via Gemini Vision — this may take a moment for large files…"
                    : "Running OCR + Gemini Vision on image…"}
                </div>
              )}

              {activeText.length > 50 && (
                <div className="doc-stats au">
                  <div className="ds"><div className="ds-n">{wordCount.toLocaleString()}</div><div className="ds-l">Words</div></div>
                  <div className="ds"><div className="ds-n">{sentCount}</div><div className="ds-l">Sentences</div></div>
                  <div className="ds"><div className="ds-n">{paraCount}</div><div className="ds-l">Paragraphs</div></div>
                  <div className="ds"><div className="ds-n">{(charCount / 1024).toFixed(1)}k</div><div className="ds-l">Characters</div></div>
                </div>
              )}

              <div className={`dropzone ${drag ? "over" : ""}`}
                onClick={() => fileRef.current?.click()}
                onDragOver={e => { e.preventDefault(); setDrag(true); }}
                onDragLeave={() => setDrag(false)}
                onDrop={e => { e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0]); }}>
                <div className="dz-center">
                  <div className="dz-icon-ring">{drag ? "🎯" : "📂"}</div>
                  <div className="dz-title">{drag ? "Drop it!" : "Drop document, PDF, or image here"}</div>
                  <div className="dz-sub">or click to browse · PDFs & images use Gemini Vision</div>
                  <div className="dz-chips">
                    {[".txt", ".md", ".csv", ".json", ".log"].map(f => (
                      <span key={f} className="dz-chip">{f}</span>
                    ))}
                    <span className="dz-chip pdf">.pdf</span>
                    {[".png", ".jpg", ".webp"].map(f => (
                      <span key={f} className="dz-chip">{f}</span>
                    ))}
                  </div>
                </div>
              </div>
              {/* ↑ accept now includes .pdf */}
              <input ref={fileRef} type="file"
                accept=".txt,.md,.csv,.json,.log,.pdf,.png,.jpg,.jpeg,.webp,.gif"
                style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])}/>

              <div className="or-bar">or paste text directly</div>
              <textarea className="paste-ta"
                placeholder="Paste lecture notes, an article, chapter text, or any study material here…"
                value={paste}
                onChange={e => { setPaste(e.target.value); setTopics([]); }}
                rows={7}/>

              {topics.length > 0 && (
                <div className="topic-card">
                  <div className="tc-hd">RAG — Detected Topics</div>
                  <div className="tc-tags">
                    {topics.map(t => <div key={t} className="tc-tag">◈ {t}</div>)}
                  </div>
                </div>
              )}

              <div className="sec-div"/>

              <div className="cfg-section">
                <div className="cfg-title">Quiz Configuration</div>
                <div className="cfg">
                  <div className="cfg-block">
                    <label className="cfg-lbl">Questions</label>
                    <select className="cfg-sel" value={numQ} onChange={e => setNumQ(e.target.value)}>
                      {["3", "5", "7", "10", "15"].map(n => <option key={n} value={n}>{n} Questions</option>)}
                    </select>
                  </div>
                  <div className="cfg-block">
                    <label className="cfg-lbl">Question Type</label>
                    <select className="cfg-sel" value={qType} onChange={e => setQType(e.target.value)}>
                      <option value="mixed">Mixed (MCQ + Short)</option>
                      <option value="mcq">Multiple Choice Only</option>
                      <option value="short">Short Answer Only</option>
                    </select>
                  </div>
                  <div className="cfg-block">
                    <label className="cfg-lbl">Difficulty</label>
                    <select className="cfg-sel" value={difficulty} onChange={e => setDiff(e.target.value)}>
                      <option value="mixed">Mixed Levels</option>
                      <option value="easy">Easy</option>
                      <option value="medium">Medium</option>
                      <option value="hard">Hard</option>
                    </select>
                  </div>
                  <div className="cfg-block">
                    <label className="cfg-lbl">Timer Mode</label>
                    <select className="cfg-sel" value={timerOn ? "on" : "off"} onChange={e => setTimerOn(e.target.value === "on")}>
                      <option value="off">No Timer</option>
                      <option value="on">Timed</option>
                    </select>
                  </div>
                  {timerOn && (
                    <div className="cfg-block">
                      <label className="cfg-lbl">Time Limit</label>
                      <select className="cfg-sel" value={timeLimit} onChange={e => setTimeL(e.target.value)}>
                        <option value="120">2 Minutes</option>
                        <option value="300">5 Minutes</option>
                        <option value="600">10 Minutes</option>
                        <option value="900">15 Minutes</option>
                      </select>
                    </div>
                  )}
                </div>
              </div>

              <div style={{ display: "flex", gap: 10, marginTop: 24, flexWrap: "wrap" }}>
                <button className="btn btn-prime" onClick={generate} disabled={!activeText.trim() || ocrActive}>
                  ⚡ Generate Quiz
                </button>
                {activeText.length > 100 && topics.length === 0 && (
                  <button className="btn btn-ghost btn-sm" onClick={extractTopics}>🔍 Extract RAG Topics</button>
                )}
              </div>
            </div>
          )}

          {/* ── GEN LOADING ── */}
          {stage === "gen-loading" && (
            <div className="load-screen au">
              <div className="loader-ring">
                <div className="lr-outer"/><div className="lr-inner"/><div className="lr-dot"/>
              </div>
              <div className="load-title">Building your quiz</div>
              <div className="load-sub">RAG pipeline → Vector retrieval → Gemini generation</div>
              <div className="load-steps">
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
              <div className="quiz-header">
                <div>
                  <div className="qh-greeting">Good luck, {userName}!</div>
                  <div className="qh-title">Answer all questions</div>
                  <div className="qh-meta">{questions.length} questions · Flag to review later · Answer at your pace</div>
                </div>
                <div className="qh-right">
                  {timerOn && (
                    <div className={`timer-widget ${timer < 60 ? "urgent" : ""}`}>
                      ⏱ {fmtTime(timer)}
                    </div>
                  )}
                  <div className="progress-chip"><strong>{answeredCount}</strong> / {questions.length} answered</div>
                </div>
              </div>

              {questions.map((q, qi) => {
                const ua        = answers[qi];
                const hasAns    = ua !== undefined && ua !== "";
                const isFlagged = flagged[qi];
                const conf      = confidence[qi];
                const dc        = diffColor[q.difficulty] || "var(--text3)";
                return (
                  <div key={qi} id={`q${qi}`}
                    className={`q-card ${hasAns ? "answered" : ""} ${isFlagged ? "flagged" : ""}`}
                    style={{ animationDelay: `${qi * 0.05}s` }}>
                    <div className="q-head">
                      <div className="q-num">{qi + 1}</div>
                      <div className="q-badge topic">{q.topic}</div>
                      <div className="q-badge" style={{ fontFamily: "'DM Mono',monospace", fontSize: "9px", padding: "3px 10px", borderRadius: "20px", background: `${dc}15`, border: `1px solid ${dc}30`, color: dc, textTransform: "uppercase", letterSpacing: "0.07em" }}>{q.difficulty}</div>
                      <div className="q-badge type">{q.type === "mcq" ? "MCQ" : "Short"}</div>
                      {hasAns && !isFlagged && <div className="q-check">Answered</div>}
                      <button className={`q-flag ${isFlagged ? "on" : ""}`} onClick={() => setFlagged(p => ({ ...p, [qi]: !p[qi] }))} title="Flag for review">🚩</button>
                    </div>
                    <div className="q-body">
                      <div className="q-text">{q.question}</div>
                      {q.type === "mcq" && (
                        <div className="opts">
                          {(q.options || []).map((opt, oi) => (
                            <button key={oi} className={`opt ${ua === opt ? "sel" : ""}`} onClick={() => setAns(p => ({ ...p, [qi]: opt }))}>
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
                              <span className="hint-ico">💡</span>
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
                <div className="res-pts">{totalPts} / {maxPts} points · Rank <strong>{rank.rank}</strong>{avgTime ? ` · Avg ${avgTime}s/q` : ""}</div>
              </div>

              <div className="stat-grid">
                <div className="stat"><div className="stat-n g">{nCorr}</div><div className="stat-l">Correct</div></div>
                <div className="stat"><div className="stat-n y">{nPart}</div><div className="stat-l">Partial</div></div>
                <div className="stat"><div className="stat-n r">{nWrong}</div><div className="stat-l">Incorrect</div></div>
                <div className="stat"><div className="stat-n p">{pct}%</div><div className="stat-l">Score</div></div>
                <div className="stat"><div className="stat-n b">{flaggedCount}</div><div className="stat-l">Flagged</div></div>
              </div>

              {/* Charts */}
              <div className="charts-section">
                <div className="section-hd">Performance Analytics · Recharts</div>
                <div className="chart-grid">
                  {radarData.length > 1 && (
                    <div className="chart-card">
                      <div className="chart-title">Topic Radar</div>
                      <ResponsiveContainer width="100%" height={220}>
                        <RadarChart data={radarData} outerRadius={80}>
                          <PolarGrid stroke="rgba(255,255,255,0.06)"/>
                          <PolarAngleAxis dataKey="topic" tick={{ fill: "#8B82B0", fontSize: 10 }}/>
                          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: "#3D3860", fontSize: 8 }}/>
                          <Radar name="Score" dataKey="score" stroke="#A78BFA" fill="#7C3AED" fillOpacity={0.25}/>
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                  <div className="chart-card">
                    <div className="chart-title">Answer Breakdown</div>
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={barData} margin={{ top: 8, right: 8, left: -20, bottom: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)"/>
                        <XAxis dataKey="name" tick={{ fill: "#8B82B0", fontSize: 11 }}/>
                        <YAxis tick={{ fill: "#3D3860", fontSize: 10 }}/>
                        <Tooltip contentStyle={{ background: "#0E0C1E", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 10, color: "#EEE9FF", fontSize: 12 }}/>
                        <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                          {barData.map((entry, i) => <rect key={i} fill={entry.fill}/>)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="chart-card full">
                    <div className="chart-title">Score by Difficulty</div>
                    <ResponsiveContainer width="100%" height={160}>
                      <BarChart data={diffData} margin={{ top: 8, right: 8, left: -20, bottom: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)"/>
                        <XAxis dataKey="diff" tick={{ fill: "#8B82B0", fontSize: 11 }}/>
                        <YAxis tick={{ fill: "#3D3860", fontSize: 10 }} domain={[0, 100]}/>
                        <Tooltip contentStyle={{ background: "#0E0C1E", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 10, color: "#EEE9FF", fontSize: 12 }} formatter={v => `${v}%`}/>
                        <Bar dataKey="pct" radius={[6, 6, 0, 0]} fill="#A78BFA"/>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Topic bars */}
              <div className="topic-breakdown">
                <div className="section-hd">Topic-wise Performance</div>
                {topicBreakdown().map((t, i) => (
                  <div key={i} className="tb-row">
                    <div className="tb-name">{t.topic}</div>
                    <div className="tb-wrap">
                      <div className="tb-fill" style={{
                        width: `${t.pct}%`,
                        background: t.pct >= 70 ? "linear-gradient(90deg,#059669,#10B981)" : t.pct >= 50 ? "linear-gradient(90deg,#D97706,#F59E0B)" : "linear-gradient(90deg,#DC2626,#F87171)",
                        animationDuration: `${0.8 + i * 0.12}s`,
                        boxShadow: t.pct >= 70 ? "0 0 8px rgba(16,185,129,0.4)" : t.pct >= 50 ? "0 0 8px rgba(245,158,11,0.4)" : "0 0 8px rgba(248,113,113,0.3)"
                      }}/>
                    </div>
                    <div className="tb-pct">{t.pct}%</div>
                  </div>
                ))}
              </div>

              {summary && (
                <div className="ai-card">
                  <div className="ai-lbl">AI Performance Summary · Gemini</div>
                  <div className="ai-txt">{summary}</div>
                </div>
              )}

              {recos.length > 0 && (
                <>
                  <div className="section-hd">AI Recommendations · Recommendation Engine</div>
                  <div className="reco-grid">
                    {recos.map((r, i) => {
                      const rs = recoStyle(r.type);
                      const ps = prioStyle(r.priority);
                      return (
                        <div key={i} className="reco-card">
                          <div className="reco-type" style={{ color: rs.color }}>
                            <span style={{ width: 6, height: 6, borderRadius: "50%", background: rs.color, display: "inline-block" }}/>
                            {r.type}
                          </div>
                          <div className="reco-topic">{r.topic}</div>
                          <div className="reco-action">{r.action}</div>
                          <div style={{ marginTop: 10 }}>
                            <span className="reco-priority" style={{ color: ps.color, background: ps.bg }}>{r.priority} priority</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}

              <div className="export-row">
                <div className="er-label">📤 Download detailed results · includes RAG stats & recommendations</div>
                <button className="btn btn-ghost btn-sm" onClick={exportResults}>Export .txt</button>
              </div>

              <div className="section-hd">Detailed Question Review</div>
              {questions.map((q, i) => {
                const fb        = feedbacks[i];
                const ua        = answers[i];
                const conf      = confidence[i];
                const isFlagged = flagged[i];
                return (
                  <div key={i} className={`rev-card ${fb?.status || ""}`} style={{ animationDelay: `${i * 0.04}s` }}>
                    <div className="rev-pills">
                      <span style={{ fontFamily: "'DM Mono',monospace", fontSize: "9px", padding: "3px 10px", borderRadius: "20px", background: "rgba(6,182,212,0.07)", border: "1px solid rgba(6,182,212,0.15)", color: "var(--cyan)", textTransform: "uppercase", letterSpacing: "0.07em" }}>{q.topic}</span>
                      {fb && <span className={`rpill ${fb.status === "correct" ? "c" : fb.status === "partial" ? "p" : "i"}`}>{fb.status === "correct" ? "✓ Correct" : fb.status === "partial" ? "◑ Partial" : "✗ Incorrect"}</span>}
                      {fb && <span className="rpill s">{fb.score}/10 pts</span>}
                      <span style={{ fontFamily: "'DM Mono',monospace", fontSize: "9px", padding: "3px 10px", borderRadius: "20px", background: `${diffColor[q.difficulty]}15`, color: diffColor[q.difficulty] }}>{q.difficulty}</span>
                      {isFlagged && <span className="rpill f">🚩 Flagged</span>}
                      {conf && <span style={{ fontFamily: "'DM Mono',monospace", fontSize: "9px", color: "var(--text3)" }}>Confidence: {conf}</span>}
                    </div>
                    <div className="rev-q">Q{i + 1}. {q.question}</div>
                    {ua && <div className="rev-ans">Your answer: <em>{ua}</em></div>}
                    {q.type === "mcq" && (
                      <div className="opts" style={{ marginTop: 10 }}>
                        {(q.options || []).map((opt, oi) => {
                          let cls = "opt";
                          if (opt === q.answer) cls += " corr";
                          else if (opt === ua && opt !== q.answer) cls += " wrong";
                          return <button key={oi} className={cls} disabled style={{ cursor: "default", fontSize: 13 }}><span className="opt-key">{LETTERS[oi]}</span>{opt}</button>;
                        })}
                      </div>
                    )}
                    {q.type === "short" && fb?.modelAns && <div className="rev-ans" style={{ marginTop: 8 }}>Model answer: <em>{fb.modelAns}</em></div>}
                    {fb && <div className="rev-fb">{fb.text}</div>}
                  </div>
                );
              })}

              {lb.length > 0 && (
                <div className="lb-card">
                  <div className="lb-hd">🏅 Session Leaderboard</div>
                  {lb.map((e, i) => (
                    <div key={i} className={`lb-row ${e.name === userName && e.pct === pct ? "me" : ""}`}>
                      <div className={`lb-pos ${i === 0 ? "g" : i === 1 ? "s" : i === 2 ? "b" : ""}`}>{i === 0 ? "👑" : i === 1 ? "🥈" : i === 2 ? "🥉" : `#${i + 1}`}</div>
                      <div className="lb-name">{e.name}</div>
                      <div className="lb-score">{e.pts}/{e.max} ({e.pct}%)</div>
                      <div className="lb-badge" style={{ color: e.color, background: e.color + "15" }}>{e.rank}</div>
                    </div>
                  ))}
                </div>
              )}

              <div className="res-acts">
                <button className="btn btn-prime" onClick={restart}>↩ New Quiz</button>
                <button className="btn btn-emerald" onClick={retake}>↺ Retake Quiz</button>
                <button className="btn btn-ghost" onClick={exportResults}>📤 Export</button>
              </div>
            </div>
          )}

        </div>

        {/* Submit bar */}
        {stage === "quiz" && (
          <div className="sub-bar">
            <div className="sb-progress">
              <div className="sb-meta"><strong>{answeredCount}</strong> of {questions.length} answered</div>
              <div className="sb-bar"><div className="sb-fill" style={{ width: `${(answeredCount / questions.length) * 100}%` }}/></div>
            </div>
            {flaggedCount > 0 && <div className="sb-flags">🚩 {flaggedCount} flagged</div>}
            <div className="sb-actions">
              {flaggedCount > 0 && (
                <button className="btn btn-ghost btn-sm" onClick={() => {
                  const fi = Object.entries(flagged).find(([k, v]) => v && !feedbacks[k]);
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
