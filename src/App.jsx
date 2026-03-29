import { useState, useRef, useEffect, useCallback } from "react";

async function gemini(prompt, system = "", maxTokens = 2048) {
  const model = "gemini-2.5-flash";
  const key = import.meta.env.VITE_GEMINI_API_KEY; // or process.env.GEMINI_API_KEY

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        system_instruction: system
          ? { parts: [{ text: system }] }
          : undefined,
        contents: [
          {
            role: "user",
            parts: [{ text: prompt }],
          },
        ],
        generationConfig: {
          maxOutputTokens: maxTokens,
          temperature: 0.75,
        },
      }),
    }
  );

  const data = await res.json();

  console.log(data);

  if (!res.ok) {
    throw new Error(data.error?.message || "Gemini API error");
  }

  return data.candidates?.[0]?.content?.parts?.[0]?.text || "";
}
// ── Helpers ──────────────────────────────────────────────────────────────────
const LETTERS = ["A","B","C","D"];
const readFileAsText = (file) =>
  new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsText(file);
  });

const getRank = (pct) => {
  if (pct >= 95) return { rank:"S+", label:"Legendary",    color:"#FFD700", bg:"rgba(255,215,0,0.08)",   emoji:"👑" };
  if (pct >= 90) return { rank:"S",  label:"Outstanding",  color:"#C084FC", bg:"rgba(192,132,252,0.08)", emoji:"⭐" };
  if (pct >= 80) return { rank:"A",  label:"Excellent",    color:"#34D399", bg:"rgba(52,211,153,0.08)",  emoji:"🏆" };
  if (pct >= 70) return { rank:"B",  label:"Good",         color:"#60A5FA", bg:"rgba(96,165,250,0.08)",  emoji:"🎯" };
  if (pct >= 60) return { rank:"C",  label:"Average",      color:"#FBBF24", bg:"rgba(251,191,36,0.08)",  emoji:"📚" };
  if (pct >= 40) return { rank:"D",  label:"Needs Work",   color:"#FB923C", bg:"rgba(251,146,60,0.08)",  emoji:"📖" };
  return               { rank:"F",  label:"Keep Trying",  color:"#F87171", bg:"rgba(248,113,113,0.08)", emoji:"💪" };
};

const formatTime = (s) => `${String(Math.floor(s/60)).padStart(2,"0")}:${String(s%60).padStart(2,"0")}`;
const diffColor = { easy:"#34D399", medium:"#FBBF24", hard:"#F87171" };

// ── Styles ───────────────────────────────────────────────────────────────────
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,300;12..96,400;12..96,500;12..96,600;12..96,700;12..96,800&family=Fira+Code:wght@300;400;500&family=Cormorant+Garamond:ital,wght@1,500;1,600&display=swap');

:root {
  --bg:      #070710;
  --bg1:     #0d0d1a;
  --bg2:     #111122;
  --bg3:     #181830;
  --glass:   rgba(255,255,255,0.03);
  --glass2:  rgba(255,255,255,0.06);
  --border:  rgba(255,255,255,0.07);
  --border2: rgba(255,255,255,0.13);
  --a1:      #7B61FF;
  --a2:      #A78BFA;
  --a3:      #C4B5FD;
  --teal:    #2DD4BF;
  --rose:    #FB7185;
  --gold:    #FFD700;
  --green:   #34D399;
  --red:     #F87171;
  --amber:   #FBBF24;
  --blue:    #60A5FA;
  --text:    #F1F1FA;
  --text2:   #9090B0;
  --text3:   #44445A;
  --r:       16px;
  --r2:      10px;
  --r3:      8px;
}

*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
html { scroll-behavior:smooth; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Bricolage Grotesque', system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
  min-height: 100vh;
  overflow-x: hidden;
}

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(123,97,255,0.3); border-radius:3px; }

/* ── Aurora BG ── */
.aurora {
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  overflow: hidden;
}
.aurora-orb {
  position: absolute; border-radius: 50%; filter: blur(100px);
  animation: drift 20s ease-in-out infinite alternate;
}
.ao1 { width:600px;height:600px; top:-200px; left:-100px; background:rgba(123,97,255,0.07); animation-duration:18s; }
.ao2 { width:500px;height:500px; top:40%; right:-150px; background:rgba(45,212,191,0.05); animation-duration:24s; animation-delay:-8s; }
.ao3 { width:400px;height:400px; bottom:-100px; left:30%; background:rgba(251,113,133,0.04); animation-duration:20s; animation-delay:-12s; }

@keyframes drift {
  0%   { transform: translate(0,0) scale(1); }
  33%  { transform: translate(40px,-30px) scale(1.1); }
  66%  { transform: translate(-20px,50px) scale(0.95); }
  100% { transform: translate(60px,20px) scale(1.05); }
}

/* Noise */
.noise {
  position: fixed; inset: 0; pointer-events: none; z-index: 0; opacity: 0.02;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  background-size: 128px;
}

.page { position:relative; z-index:1; min-height:100vh; }

/* ── Animations ── */
@keyframes up      { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn  { from{opacity:0} to{opacity:1} }
@keyframes spin    { to{transform:rotate(360deg)} }
@keyframes pulse   { 0%,100%{opacity:1} 50%{opacity:0.3} }
@keyframes scanline{ from{transform:translateY(-100%)} to{transform:translateY(100vh)} }
@keyframes ticker  { 0%,100%{box-shadow:0 0 12px var(--a1)} 50%{box-shadow:0 0 30px var(--a1), 0 0 60px rgba(123,97,255,0.3)} }
@keyframes popIn   { 0%{transform:scale(0.6);opacity:0} 70%{transform:scale(1.08)} 100%{transform:scale(1);opacity:1} }
@keyframes slideR  { from{transform:translateX(-16px);opacity:0} to{transform:translateX(0);opacity:1} }
@keyframes barFill { from{width:0} }
@keyframes countUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
@keyframes flagWave{ 0%,100%{transform:rotate(-3deg)} 50%{transform:rotate(3deg)} }

.a-up   { animation: up    0.5s cubic-bezier(0.16,1,0.3,1) both; }
.a-up2  { animation: up    0.5s cubic-bezier(0.16,1,0.3,1) 0.08s both; }
.a-up3  { animation: up    0.5s cubic-bezier(0.16,1,0.3,1) 0.16s both; }
.a-pop  { animation: popIn 0.6s cubic-bezier(0.34,1.56,0.64,1) both; }
.a-fade { animation: fadeIn 0.4s ease both; }

/* ── NAV ── */
.nav {
  display:flex; align-items:center; gap:12px;
  padding:16px 40px;
  border-bottom: 1px solid var(--border);
  background: rgba(7,7,16,0.8);
  backdrop-filter: blur(24px);
  position: sticky; top:0; z-index:200;
}
.nav-logo {
  display:flex; align-items:center; gap:10px;
  font-size:18px; font-weight:800; letter-spacing:-0.5px;
}
.nav-logo-mark {
  width:34px; height:34px; border-radius:10px;
  background: linear-gradient(135deg, var(--a1), var(--teal));
  display:flex; align-items:center; justify-content:center;
  font-size:16px;
  box-shadow: 0 0 20px rgba(123,97,255,0.4);
  animation: ticker 3s ease infinite;
}
.nav-logo-txt em { font-style:normal; color:var(--a2); }
.nav-chip {
  font-family:'Fira Code', monospace; font-size:10px;
  padding:3px 10px; border-radius:20px;
  background:rgba(123,97,255,0.1); border:1px solid rgba(123,97,255,0.25);
  color:var(--a3); letter-spacing:0.04em;
}
.nav-r { margin-left:auto; display:flex; align-items:center; gap:16px; }
.nav-steps { display:flex; align-items:center; gap:3px; }
.ns {
  font-family:'Fira Code', monospace; font-size:9px;
  padding:4px 12px; border-radius:20px; letter-spacing:0.06em;
  text-transform:uppercase; border:1px solid var(--border); color:var(--text3);
  transition:all 0.25s;
}
.ns.on { background:rgba(123,97,255,0.15); border-color:rgba(123,97,255,0.35); color:var(--a2); box-shadow:0 0 12px rgba(123,97,255,0.2); }
.ns.done { background:rgba(52,211,153,0.1); border-color:rgba(52,211,153,0.3); color:var(--green); }

/* ── Container ── */
.wrap { max-width:820px; margin:0 auto; padding:52px 28px 120px; }

/* ── Buttons ── */
.btn {
  padding:13px 26px; border-radius:var(--r2);
  font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:14px;
  cursor:pointer; transition:all 0.2s; display:inline-flex; align-items:center; gap:9px;
  border:none; outline:none; letter-spacing:0.01em;
}
.btn-prime {
  background: linear-gradient(135deg, var(--a1) 0%, #9333EA 100%);
  color:#fff; box-shadow:0 4px 20px rgba(123,97,255,0.35);
}
.btn-prime:hover:not(:disabled) { transform:translateY(-2px); box-shadow:0 8px 32px rgba(123,97,255,0.5); }
.btn-prime:active:not(:disabled) { transform:translateY(0); }
.btn-prime:disabled { opacity:0.3; cursor:not-allowed; }
.btn-teal {
  background: linear-gradient(135deg, var(--teal), #0D9488);
  color:#fff; box-shadow:0 4px 20px rgba(45,212,191,0.25);
}
.btn-teal:hover:not(:disabled) { transform:translateY(-2px); box-shadow:0 8px 28px rgba(45,212,191,0.4); }
.btn-ghost {
  background:var(--glass2); color:var(--text2);
  border:1px solid var(--border2);
}
.btn-ghost:hover { background:rgba(255,255,255,0.08); color:var(--text); }
.btn-danger {
  background:rgba(248,113,113,0.1); color:var(--red);
  border:1px solid rgba(248,113,113,0.2);
}
.btn-danger:hover { background:rgba(248,113,113,0.18); }
.btn-sm { padding:9px 18px; font-size:13px; }

/* ── ─── NAME SCREEN ─── ── */
.hero {
  display:flex; flex-direction:column; align-items:center;
  text-align:center; padding:72px 0 56px;
}
.hero-badge {
  display:inline-flex; align-items:center; gap:8px; margin-bottom:28px;
  font-family:'Fira Code', monospace; font-size:11px; letter-spacing:0.1em;
  text-transform:uppercase; color:var(--a2);
  padding:7px 18px; border-radius:40px;
  border:1px solid rgba(123,97,255,0.25); background:rgba(123,97,255,0.07);
}
.hb-dot { width:6px;height:6px;border-radius:50%;background:var(--a1);animation:pulse 2s ease infinite; }
.hero-h1 {
  font-size:clamp(48px,7vw,80px); font-weight:800;
  line-height:0.95; letter-spacing:-3px; margin-bottom:20px;
}
.hero-h1 .w1 { display:block; color:var(--text); }
.hero-h1 .w2 {
  display:block; font-family:'Cormorant Garamond',serif; font-style:italic;
  font-size:clamp(52px,8vw,90px);
  background:linear-gradient(120deg, var(--a2) 0%, var(--teal) 50%, var(--gold) 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  letter-spacing:-2px;
}
.hero-p { font-size:16px; line-height:1.75; color:var(--text2); max-width:500px; margin-bottom:52px; }

.name-glass {
  width:100%; max-width:440px;
  background:var(--glass); border:1px solid var(--border2);
  border-radius:20px; padding:36px 32px;
  box-shadow:0 40px 80px rgba(0,0,0,0.4), 0 0 0 1px rgba(123,97,255,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
  backdrop-filter:blur(20px);
  animation: up 0.6s cubic-bezier(0.16,1,0.3,1) 0.15s both;
}
.ng-lbl { font-family:'Fira Code',monospace; font-size:10px; letter-spacing:0.12em; text-transform:uppercase; color:var(--text3); margin-bottom:10px; display:block; }
.ng-input {
  width:100%; padding:15px 20px; margin-bottom:22px;
  border:1px solid var(--border2); border-radius:12px;
  background:rgba(0,0,0,0.3); color:var(--text);
  font-family:'Bricolage Grotesque',sans-serif; font-size:20px; font-weight:600;
  outline:none; transition:all 0.25s; text-align:center;
}
.ng-input:focus { border-color:rgba(123,97,255,0.5); box-shadow:0 0 0 3px rgba(123,97,255,0.12), 0 0 40px rgba(123,97,255,0.08); }
.ng-input::placeholder { color:var(--text3); font-weight:400; }

.feature-pills { display:flex; gap:8px; flex-wrap:wrap; justify-content:center; margin-top:36px; }
.fpill {
  display:flex; align-items:center; gap:6px;
  font-size:12px; font-weight:600; color:var(--text3);
  padding:6px 14px; border-radius:40px;
  border:1px solid var(--border); background:var(--glass);
}
.fpill-ico { font-size:14px; }

/* ── UPLOAD ── */
.up-greeting {
  font-family:'Fira Code',monospace; font-size:12px; color:var(--teal);
  letter-spacing:0.06em; margin-bottom:8px; display:flex; align-items:center; gap:8px;
}
.up-title { font-size:34px; font-weight:800; letter-spacing:-1px; margin-bottom:8px; }
.up-sub { font-size:15px; color:var(--text2); margin-bottom:32px; line-height:1.65; }

.dropzone {
  border:2px dashed rgba(123,97,255,0.2); border-radius:var(--r);
  padding:52px 36px; text-align:center; cursor:pointer;
  transition:all 0.25s; position:relative; overflow:hidden;
  background:linear-gradient(135deg, rgba(123,97,255,0.02), rgba(45,212,191,0.01));
}
.dropzone::before {
  content:''; position:absolute; inset:0;
  background:radial-gradient(ellipse at center, rgba(123,97,255,0.07) 0%, transparent 70%);
  opacity:0; transition:opacity 0.3s;
}
.dropzone:hover::before, .dropzone.over::before { opacity:1; }
.dropzone:hover, .dropzone.over {
  border-color:rgba(123,97,255,0.45);
  box-shadow:0 20px 60px rgba(0,0,0,0.3), 0 0 50px rgba(123,97,255,0.07);
  transform:translateY(-3px);
}
.dz-ico-wrap {
  width:64px;height:64px; margin:0 auto 18px; border-radius:18px;
  background:linear-gradient(135deg,rgba(123,97,255,0.15),rgba(45,212,191,0.1));
  border:1px solid rgba(123,97,255,0.2);
  display:flex; align-items:center; justify-content:center;
  font-size:28px; transition:all 0.25s; position:relative; z-index:1;
}
.dropzone:hover .dz-ico-wrap { transform:scale(1.1) rotate(-3deg); box-shadow:0 0 30px rgba(123,97,255,0.35); }
.dz-title { font-weight:700; font-size:17px; margin-bottom:4px; position:relative;z-index:1; }
.dz-sub { font-family:'Fira Code',monospace; font-size:12px; color:var(--text3); position:relative;z-index:1; }
.dz-types { display:flex; gap:6px; justify-content:center; margin-top:14px; flex-wrap:wrap; position:relative;z-index:1; }
.dz-type {
  font-family:'Fira Code',monospace; font-size:10px; padding:3px 10px; border-radius:20px;
  background:rgba(123,97,255,0.08); border:1px solid var(--border); color:var(--text3); letter-spacing:0.05em;
}

.file-pill {
  display:flex; align-items:center; gap:14px;
  padding:14px 18px; border-radius:var(--r);
  border:1px solid rgba(123,97,255,0.25); background:rgba(123,97,255,0.05);
  margin-bottom:16px; animation:up 0.3s ease both;
}
.fp-ico { width:44px;height:44px; border-radius:12px; background:linear-gradient(135deg,rgba(123,97,255,0.25),rgba(45,212,191,0.15)); display:flex;align-items:center;justify-content:center; font-size:22px; }
.fp-info { flex:1; }
.fp-name { font-weight:700; font-size:14px; }
.fp-meta { font-family:'Fira Code',monospace; font-size:11px; color:var(--text3); margin-top:3px; }
.fp-rm { width:30px;height:30px; border-radius:8px; border:1px solid var(--border); background:transparent; color:var(--text3); cursor:pointer; font-size:14px; display:flex;align-items:center;justify-content:center; transition:all 0.2s; }
.fp-rm:hover { border-color:var(--red); color:var(--red); background:rgba(248,113,113,0.08); }

.or-line { display:flex; align-items:center; gap:14px; margin:22px 0; font-family:'Fira Code',monospace; font-size:11px; color:var(--text3); }
.or-line::before,.or-line::after { content:''; flex:1; height:1px; background:var(--border); }

.paste-area {
  width:100%; min-height:150px; padding:18px 20px;
  border:1px solid var(--border); border-radius:var(--r);
  background:var(--bg1); resize:vertical;
  font-family:'Bricolage Grotesque',sans-serif; font-size:15px; line-height:1.7; color:var(--text);
  outline:none; transition:all 0.2s;
}
.paste-area:focus { border-color:rgba(123,97,255,0.4); box-shadow:0 0 0 3px rgba(123,97,255,0.08); }
.paste-area::placeholder { color:var(--text3); }

/* Doc preview / stats */
.doc-stats {
  display:grid; grid-template-columns:repeat(4,1fr); gap:1px;
  background:var(--border); border:1px solid var(--border); border-radius:var(--r);
  overflow:hidden; margin-bottom:24px; animation:up 0.4s ease both;
}
.ds { background:var(--bg1); padding:16px; text-align:center; }
.ds-n { font-size:22px; font-weight:800; color:var(--a2); letter-spacing:-0.5px; }
.ds-l { font-family:'Fira Code',monospace; font-size:9px; color:var(--text3); text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }

/* Config */
.cfg { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:20px; }
.cfg-lbl { display:block; font-family:'Fira Code',monospace; font-size:10px; text-transform:uppercase; letter-spacing:0.12em; color:var(--text3); margin-bottom:8px; }
.cfg-sel {
  width:100%; padding:12px 16px; border:1px solid var(--border); border-radius:var(--r3);
  background:var(--bg1); color:var(--text);
  font-family:'Bricolage Grotesque',sans-serif; font-size:14px; font-weight:600;
  outline:none; cursor:pointer; transition:all 0.2s; appearance:none;
}
.cfg-sel:focus { border-color:rgba(123,97,255,0.4); box-shadow:0 0 0 3px rgba(123,97,255,0.08); }

/* Topic preview */
.topic-preview {
  margin-top:24px; padding:20px 22px; border-radius:var(--r);
  border:1px solid rgba(45,212,191,0.2);
  background:rgba(45,212,191,0.03);
  animation:up 0.4s ease both;
}
.tp-label { font-family:'Fira Code',monospace; font-size:10px; text-transform:uppercase; letter-spacing:0.1em; color:var(--teal); margin-bottom:12px; display:flex; align-items:center; gap:8px; }
.tp-tags { display:flex; flex-wrap:wrap; gap:7px; }
.tp-tag {
  font-size:13px; font-weight:600; padding:5px 14px; border-radius:20px;
  background:rgba(45,212,191,0.08); border:1px solid rgba(45,212,191,0.15);
  color:var(--teal); display:flex; align-items:center; gap:6px;
}

/* ── LOADING ── */
.load-screen { text-align:center; padding:100px 24px; }
.spinner {
  width:60px;height:60px; margin:0 auto 28px;
  border:2px solid var(--border2);
  border-top-color:var(--a1); border-right-color:rgba(123,97,255,0.5);
  border-radius:50%; animation:spin 0.85s linear infinite;
  box-shadow:0 0 30px rgba(123,97,255,0.2);
}
.load-title { font-size:28px; font-weight:800; letter-spacing:-0.8px; margin-bottom:8px; }
.load-sub { font-size:15px; color:var(--text2); }
.load-list { margin-top:32px; display:flex; flex-direction:column; gap:10px; align-items:center; }
.ll-item { font-family:'Fira Code',monospace; font-size:11px; color:var(--text3); display:flex;align-items:center;gap:10px; animation:up 0.4s ease both; }
.ll-dot { width:5px;height:5px;border-radius:50%;background:var(--a1);animation:pulse 1.2s ease infinite; }

/* ── QUIZ ── */
.quiz-top {
  display:flex; align-items:flex-start; justify-content:space-between;
  margin-bottom:32px; gap:16px; flex-wrap:wrap;
}
.qt-left {}
.qt-greet { font-family:'Fira Code',monospace; font-size:11px; color:var(--teal); margin-bottom:6px; letter-spacing:0.06em; }
.qt-title { font-size:30px; font-weight:800; letter-spacing:-0.8px; margin-bottom:4px; }
.qt-sub { font-size:14px; color:var(--text3); }
.qt-right { display:flex; flex-direction:column; align-items:flex-end; gap:8px; }
.qt-timer {
  font-family:'Fira Code',monospace; font-size:22px; font-weight:500;
  color:var(--amber); letter-spacing:0.02em;
  padding:8px 16px; border-radius:10px;
  background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.2);
  display:flex; align-items:center; gap:8px;
}
.qt-timer.urgent { color:var(--red); background:rgba(248,113,113,0.08); border-color:rgba(248,113,113,0.25); animation:pulse 1s ease infinite; }
.qt-counter {
  font-family:'Fira Code',monospace; font-size:11px; color:var(--text2);
  padding:6px 14px; border-radius:20px;
  background:rgba(123,97,255,0.08); border:1px solid rgba(123,97,255,0.2);
}
.qt-counter strong { color:var(--a2); }

/* Question card */
.q-card {
  border-radius:18px; border:1px solid var(--border);
  background:var(--bg1); overflow:hidden; margin-bottom:18px;
  transition:border-color 0.3s, box-shadow 0.3s;
  animation: up 0.5s cubic-bezier(0.16,1,0.3,1) both;
}
.q-card:hover { border-color:var(--border2); }
.q-card.answered { border-color:rgba(123,97,255,0.2); box-shadow:0 0 20px rgba(123,97,255,0.04); }
.q-card.flagged { border-color:rgba(251,191,36,0.3); box-shadow:0 0 20px rgba(251,191,36,0.05); }

.q-head {
  display:flex; align-items:center; gap:12px; padding:18px 22px 14px;
  border-bottom:1px solid var(--border);
}
.q-num {
  width:36px;height:36px; border-radius:10px; flex-shrink:0;
  display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,rgba(123,97,255,0.2),rgba(45,212,191,0.1));
  border:1px solid rgba(123,97,255,0.2);
  font-size:14px; font-weight:800; color:var(--a2);
}
.q-topic {
  font-family:'Fira Code',monospace; font-size:10px;
  padding:3px 10px; border-radius:20px;
  background:rgba(45,212,191,0.07); border:1px solid rgba(45,212,191,0.15);
  color:var(--teal); letter-spacing:0.06em; text-transform:uppercase;
}
.q-diff {
  font-family:'Fira Code',monospace; font-size:10px;
  padding:3px 10px; border-radius:20px; letter-spacing:0.06em; text-transform:uppercase;
}
.q-type-chip {
  font-family:'Fira Code',monospace; font-size:10px;
  padding:3px 10px; border-radius:20px;
  background:rgba(251,191,36,0.07); border:1px solid rgba(251,191,36,0.15);
  color:var(--amber); letter-spacing:0.06em; text-transform:uppercase;
}
.q-flag-btn {
  margin-left:auto; width:32px;height:32px; border-radius:8px;
  border:1px solid var(--border); background:transparent; cursor:pointer;
  color:var(--text3); font-size:15px; transition:all 0.2s;
  display:flex;align-items:center;justify-content:center;
}
.q-flag-btn:hover { border-color:rgba(251,191,36,0.4); color:var(--amber); }
.q-flag-btn.active { border-color:rgba(251,191,36,0.4); background:rgba(251,191,36,0.08); color:var(--amber); animation:flagWave 0.5s ease; }
.q-answered-mark { margin-left:auto; font-family:'Fira Code',monospace; font-size:10px; color:var(--teal); }

.q-body { padding:20px 22px; }
.q-text { font-size:17px; font-weight:700; line-height:1.5; margin-bottom:20px; letter-spacing:-0.2px; }

/* Confidence */
.confidence-row { display:flex; align-items:center; gap:8px; margin-top:16px; }
.conf-lbl { font-family:'Fira Code',monospace; font-size:10px; color:var(--text3); text-transform:uppercase; letter-spacing:0.08em; white-space:nowrap; }
.conf-btn {
  padding:5px 12px; border-radius:20px; border:1px solid var(--border); background:transparent;
  font-size:12px; font-weight:600; cursor:pointer; transition:all 0.18s; color:var(--text3);
}
.conf-btn:hover { border-color:var(--border2); color:var(--text); }
.conf-btn.active { color:#fff; border-color:transparent; }
.conf-high  { background:rgba(52,211,153,0.15);  border-color:rgba(52,211,153,0.3);  color:var(--green) !important; }
.conf-high.active  { background:var(--green); color:#fff !important; }
.conf-med   { background:rgba(251,191,36,0.1);   border-color:rgba(251,191,36,0.25); color:var(--amber) !important; }
.conf-med.active   { background:var(--amber); color:#fff !important; }
.conf-low   { background:rgba(248,113,113,0.1);  border-color:rgba(248,113,113,0.25);color:var(--red) !important; }
.conf-low.active   { background:var(--red); color:#fff !important; }

/* Options */
.opts { display:flex; flex-direction:column; gap:9px; }
.opt {
  display:flex; align-items:flex-start; gap:13px; padding:13px 16px;
  border-radius:12px; border:1px solid var(--border); background:var(--bg2);
  cursor:pointer; text-align:left; font-size:14px; line-height:1.55;
  transition:all 0.15s; color:var(--text); width:100%;
  font-family:'Bricolage Grotesque',sans-serif; font-weight:500;
}
.opt:hover:not(:disabled) { border-color:rgba(123,97,255,0.35); background:rgba(123,97,255,0.07); transform:translateX(4px); }
.opt.sel   { border-color:rgba(123,97,255,0.45); background:rgba(123,97,255,0.09); }
.opt.corr  { border-color:rgba(52,211,153,0.45);  background:rgba(52,211,153,0.07); }
.opt.wrong { border-color:rgba(248,113,113,0.35); background:rgba(248,113,113,0.07); }
.opt:disabled { cursor:default; }
.opt-k {
  width:26px;height:26px; border-radius:8px; flex-shrink:0;
  display:flex;align-items:center;justify-content:center;
  border:1px solid var(--border);
  font-family:'Fira Code',monospace; font-size:11px; font-weight:500;
  transition:all 0.15s; margin-top:1px;
}
.opt.sel   .opt-k { border-color:var(--a1); background:var(--a1); color:#fff; }
.opt.corr  .opt-k { border-color:var(--green); background:var(--green); color:#fff; }
.opt.wrong .opt-k { border-color:var(--red); background:var(--red); color:#fff; }

.short-ta {
  width:100%; min-height:115px; padding:14px 18px;
  border:1px solid var(--border); border-radius:12px;
  background:var(--bg2); resize:vertical;
  font-family:'Bricolage Grotesque',sans-serif; font-size:15px; line-height:1.65; color:var(--text);
  outline:none; transition:all 0.2s;
}
.short-ta:focus { border-color:rgba(123,97,255,0.4); box-shadow:0 0 0 3px rgba(123,97,255,0.08); }
.short-ta:disabled { background:var(--bg3); color:var(--text2); }

/* ── Submit bar ── */
.sub-bar {
  position:fixed; bottom:0; left:0; right:0; z-index:100;
  background:rgba(7,7,16,0.92); backdrop-filter:blur(24px);
  border-top:1px solid var(--border);
  padding:16px 40px;
  display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
}
.sb-progress { flex:1; display:flex; flex-direction:column; gap:6px; min-width:120px; }
.sb-prog-bar { height:4px; background:var(--bg3); border-radius:2px; overflow:hidden; }
.sb-prog-inner { height:100%; background:linear-gradient(90deg,var(--a1),var(--teal)); border-radius:2px; transition:width 0.4s ease; animation:barFill 0.5s ease; }
.sb-prog-label { font-family:'Fira Code',monospace; font-size:10px; color:var(--text3); }
.sb-prog-label strong { color:var(--a2); }
.sb-flags { font-family:'Fira Code',monospace; font-size:11px; color:var(--amber); display:flex;align-items:center;gap:6px; }
.sb-mid { display:flex; gap:8px; }

/* ── EVALUATE overlay ── */
.eval-overlay {
  position:fixed; inset:0; z-index:300;
  background:rgba(7,7,16,0.95); backdrop-filter:blur(16px);
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  animation:fadeIn 0.4s ease both;
}
.eval-spinner {
  width:72px;height:72px; margin:0 auto 28px;
  border:3px solid var(--border2);
  border-top-color:var(--a1); border-right-color:rgba(45,212,191,0.6);
  border-radius:50%; animation:spin 0.85s linear infinite;
  box-shadow:0 0 50px rgba(123,97,255,0.25);
}
.eval-title { font-size:26px; font-weight:800; margin-bottom:8px; letter-spacing:-0.5px; }
.eval-sub { font-size:14px; color:var(--text2); margin-bottom:32px; }
.eval-progress { width:360px; max-width:90vw; }
.eval-bar-wrap { height:6px; background:var(--bg3); border-radius:3px; overflow:hidden; margin-bottom:10px; }
.eval-bar { height:100%; background:linear-gradient(90deg,var(--a1),var(--teal)); border-radius:3px; transition:width 0.6s ease; }
.eval-step-txt { font-family:'Fira Code',monospace; font-size:11px; color:var(--text3); text-align:center; }

/* ── RESULTS ── */
.results-hero { text-align:center; padding:60px 24px 48px; position:relative; }
.rank-orb {
  display:inline-flex; flex-direction:column; align-items:center; justify-content:center;
  width:148px;height:148px; border-radius:50%; margin:0 auto 28px;
  border:2px solid currentColor;
  box-shadow:0 0 80px currentColor, inset 0 0 40px rgba(0,0,0,0.5);
  position:relative;
  animation:popIn 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.2s both;
}
.rank-orb::before {
  content:''; position:absolute; inset:-3px; border-radius:50%;
  border:1px solid currentColor; opacity:0.3;
  animation:spin 6s linear infinite;
  background:conic-gradient(from 0, transparent 0%, currentColor 15%, transparent 30%);
}
.rank-orb::after {
  content:''; position:absolute; inset:-8px; border-radius:50%;
  border:1px solid currentColor; opacity:0.1;
}
.rank-letter { font-size:52px; font-weight:800; line-height:1; letter-spacing:-2px; position:relative;z-index:1; }
.rank-pct { font-family:'Fira Code',monospace; font-size:13px; opacity:0.7; margin-top:2px; position:relative;z-index:1; }
.res-emoji { font-size:36px; margin-bottom:10px; display:block; }
.res-user { font-family:'Fira Code',monospace; font-size:13px; color:var(--text3); margin-bottom:8px; letter-spacing:0.04em; }
.res-label { font-size:38px; font-weight:800; letter-spacing:-1.2px; margin-bottom:8px; }
.res-pts { font-size:16px; color:var(--text2); }

/* Stats */
.stat-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:1px; background:var(--border); border:1px solid var(--border); border-radius:var(--r); overflow:hidden; margin:36px 0; }
.stat { background:var(--bg1); padding:20px 12px; text-align:center; }
.stat-n { font-size:28px; font-weight:800; letter-spacing:-0.5px; line-height:1; }
.stat-n.g{color:var(--green)} .stat-n.y{color:var(--amber)} .stat-n.r{color:var(--red)} .stat-n.p{color:var(--a2)} .stat-n.b{color:var(--blue)}
.stat-l { font-family:'Fira Code',monospace; font-size:9px; color:var(--text3); text-transform:uppercase; letter-spacing:0.08em; margin-top:5px; }

/* Topic breakdown */
.topic-breakdown { margin-bottom:36px; }
.tb-title { font-family:'Fira Code',monospace; font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:var(--text3); margin-bottom:16px; display:flex;align-items:center;gap:10px; }
.tb-title::after { content:''; flex:1; height:1px; background:var(--border); }
.tb-row { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
.tb-name { font-size:13px; font-weight:600; min-width:140px; color:var(--text2); }
.tb-bar-wrap { flex:1; height:8px; background:var(--bg3); border-radius:4px; overflow:hidden; }
.tb-bar { height:100%; border-radius:4px; animation:barFill 1s ease; }
.tb-pct { font-family:'Fira Code',monospace; font-size:11px; color:var(--text3); min-width:36px; text-align:right; }

/* AI Summary */
.ai-box {
  padding:28px 32px; border-radius:var(--r);
  border:1px solid var(--border2);
  background:linear-gradient(135deg,rgba(123,97,255,0.05),rgba(45,212,191,0.03));
  margin-bottom:36px; position:relative; overflow:hidden;
}
.ai-box::before {
  content:'✦'; position:absolute; bottom:-10px; right:20px;
  font-size:80px; opacity:0.03; color:var(--a2); line-height:1;
}
.ai-box-lbl { font-family:'Fira Code',monospace; font-size:10px; text-transform:uppercase; letter-spacing:0.12em; color:var(--a2); margin-bottom:10px; display:flex;align-items:center;gap:8px; }
.ai-box-lbl::before { content:''; display:block; width:20px;height:1px;background:var(--a2); }
.ai-box-txt { font-size:15px; line-height:1.82; color:var(--text2); }

/* Review */
.review-section-hd { font-family:'Fira Code',monospace; font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:var(--text3); margin-bottom:16px; display:flex;align-items:center;gap:10px; }
.review-section-hd::after { content:''; flex:1; height:1px; background:var(--border); }

.rev {
  margin-bottom:14px; padding:20px 22px; border-radius:var(--r);
  border:1px solid var(--border); background:var(--bg1); animation:up 0.4s ease both;
}
.rev.correct   { border-color:rgba(52,211,153,0.22); }
.rev.incorrect { border-color:rgba(248,113,113,0.18); }
.rev.partial   { border-color:rgba(251,191,36,0.22); }
.rev-pills { display:flex; align-items:center; gap:7px; flex-wrap:wrap; margin-bottom:10px; }
.rpill { font-family:'Fira Code',monospace; font-size:10px; padding:2px 10px; border-radius:20px; text-transform:uppercase; letter-spacing:0.05em; }
.rpill.c { background:rgba(52,211,153,0.1);  color:var(--green); }
.rpill.i { background:rgba(248,113,113,0.1); color:var(--red); }
.rpill.p { background:rgba(251,191,36,0.1);  color:var(--amber); }
.rpill.s { background:rgba(123,97,255,0.1);  color:var(--a2); }
.rpill.f { background:rgba(251,191,36,0.08); color:var(--amber); border:1px solid rgba(251,191,36,0.2); }
.rev-q { font-size:14px; font-weight:700; margin-bottom:8px; line-height:1.4; }
.rev-ans { font-size:13px; color:var(--text3); margin-bottom:3px; }
.rev-ans span { color:var(--text2); font-style:italic; }
.rev-opts { display:flex; flex-direction:column; gap:7px; margin-top:10px; }
.rev-fb { font-size:13px; color:var(--text3); line-height:1.65; margin-top:10px; padding-top:10px; border-top:1px solid var(--border); }
.rev-conf { font-family:'Fira Code',monospace; font-size:10px; color:var(--text3); }

/* Export bar */
.export-bar {
  display:flex; align-items:center; gap:10px; margin-top:10px; margin-bottom:24px;
  padding:14px 18px; border-radius:var(--r);
  background:var(--glass); border:1px solid var(--border);
  flex-wrap:wrap;
}
.eb-label { font-family:'Fira Code',monospace; font-size:11px; color:var(--text3); flex:1; }

/* Leaderboard */
.leaderboard { margin-top:32px; padding:24px; border-radius:var(--r); border:1px solid var(--border); background:var(--bg1); }
.lb-title { font-family:'Fira Code',monospace; font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:var(--text3); margin-bottom:16px; }
.lb-row { display:flex; align-items:center; gap:12px; padding:10px 12px; border-radius:10px; margin-bottom:6px; background:var(--glass); }
.lb-pos { font-family:'Fira Code',monospace; font-size:13px; font-weight:700; width:24px; color:var(--text3); }
.lb-pos.first { color:var(--gold); }
.lb-pos.second { color:#C0C0C0; }
.lb-pos.third { color:#CD7F32; }
.lb-name { flex:1; font-weight:700; font-size:14px; }
.lb-score { font-family:'Fira Code',monospace; font-size:12px; color:var(--a2); }
.lb-rank { font-size:12px; font-weight:700; padding:2px 8px; border-radius:6px; }

.res-acts { display:flex; gap:10px; margin-top:32px; flex-wrap:wrap; }

/* ── Toast ── */
.toast {
  position:fixed; bottom:88px; right:24px; z-index:400;
  padding:13px 20px; border-radius:var(--r2);
  background:var(--bg2); border:1px solid var(--border2);
  color:var(--text); font-family:'Fira Code',monospace; font-size:12px;
  box-shadow:0 20px 60px rgba(0,0,0,0.5); animation:up 0.3s ease both;
  max-width:320px; display:flex;align-items:center;gap:10px;
}

/* ── Section divider ── */
.sec-div { height:1px; background:var(--border); margin:28px 0; }

@media(max-width:640px){
  .nav{padding:12px 16px}
  .nav-steps{display:none}
  .wrap{padding:32px 16px 120px}
  .stat-grid{grid-template-columns:repeat(3,1fr)}
  .cfg{grid-template-columns:1fr}
  .doc-stats{grid-template-columns:repeat(2,1fr)}
  .quiz-top{flex-direction:column}
  .sub-bar{padding:12px 16px}
}
`;

// ── LEADERBOARD (session memory) ────────────────────────────────────────────
let LEADERBOARD = JSON.parse(sessionStorage.getItem("quiz_lb") || "[]");
const saveLB = (entry) => {
  LEADERBOARD = [entry, ...LEADERBOARD].slice(0, 8)
    .sort((a, b) => b.pct - a.pct);
  sessionStorage.setItem("quiz_lb", JSON.stringify(LEADERBOARD));
};

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [stage, setStage]       = useState("name");
  const [userName, setUser]     = useState("");
  const [docText, setDocText]   = useState("");
  const [paste, setPaste]       = useState("");
  const [fileName, setFile]     = useState("");
  const [numQ, setNumQ]         = useState("5");
  const [qType, setQType]       = useState("mixed");
  const [difficulty, setDiff]   = useState("mixed");
  const [timerOn, setTimerOn]   = useState(false);
  const [timeLimit, setTimeL]   = useState("300");
  const [drag, setDrag]         = useState(false);
  const [genStep, setGenStep]   = useState(0);
  const [evalPct, setEvalPct]   = useState(0);
  const [evalTxt, setEvalTxt]   = useState("");
  const [questions, setQ]       = useState([]);
  const [topics, setTopics]     = useState([]);
  const [answers, setAns]       = useState({});
  const [confidence, setConf]   = useState({});
  const [flagged, setFlagged]   = useState({});
  const [feedbacks, setFb]      = useState({});
  const [summary, setSummary]   = useState("");
  const [timer, setTimer]       = useState(0);
  const [timerActive, setTA]    = useState(false);
  const [toast, setToast]       = useState("");
  const [lb, setLB]             = useState(LEADERBOARD);
  const fileRef = useRef();
  const timerRef = useRef();

  const GEN_STEPS = ["Ingesting document…","Extracting key topics…","Generating questions…","Calibrating difficulty…","Finalising quiz…"];

  const msg = (m, icon="ℹ") => { setToast({ m, icon }); setTimeout(() => setToast(""), 3800); };

  // ── Timer ──
  useEffect(() => {
    if (timerActive && timer > 0) {
      timerRef.current = setTimeout(() => setTimer(t => t - 1), 1000);
    } else if (timerActive && timer === 0) {
      msg("⏱ Time's up! Submitting…","⏰");
      evaluateAll();
    }
    return () => clearTimeout(timerRef.current);
  }, [timer, timerActive]);

  // ── File ──
  const handleFile = async (f) => {
    if (!f) return;
    try { const t = await readFileAsText(f); setDocText(t); setFile(f.name); }
    catch { msg("Could not read file. Try .txt / .md / .csv","❌"); }
  };

  // ── Doc stats ──
  const activeText = docText || paste;
  const wordCount = activeText ? activeText.split(/\s+/).filter(Boolean).length : 0;
  const sentCount = activeText ? activeText.split(/[.!?]+/).filter(Boolean).length : 0;
  const paraCount = activeText ? activeText.split(/\n\n+/).filter(Boolean).length : 0;
  const charCount = activeText.length;

  // ── Extract topics preview ──
  const extractTopics = async () => {
    const text = activeText.trim();
    if (text.length < 80) return;
    try {
      const raw = await gemini(
        `Extract 5-8 key topics from this document. Return ONLY a JSON array of short strings (2-4 words each). Example: ["Machine Learning","Neural Networks"]\n\nDocument:\n${text.slice(0,3000)}`,
        "You are a document analyst. Return only valid JSON arrays.", 200
      );
      const arr = JSON.parse(raw.replace(/```json|```/gi,"").trim());
      if (Array.isArray(arr)) setTopics(arr);
    } catch {}
  };

  useEffect(() => {
    if (activeText.length > 100 && topics.length === 0) {
      const t = setTimeout(extractTopics, 1200);
      return () => clearTimeout(t);
    }
  }, [activeText]);

  // ── Generate ──
  const generate = async () => {
    const text = activeText.trim();
    if (!text || text.length < 50) { msg("Please provide at least 50 characters","⚠"); return; }
    setStage("gen-loading"); setGenStep(0);
    const iv = setInterval(() => setGenStep(s => Math.min(s + 1, GEN_STEPS.length - 1)), 850);

    const sys = `You are an expert quiz generator and educator. Return ONLY a valid JSON array.
Schema per item:
{ "id":number, "type":"mcq"|"short", "topic":string, "difficulty":"easy"|"medium"|"hard",
  "question":string,
  MCQ: "options":[4 strings], "answer":string (exact match), "explanation":string,
  Short: "answer":string (model answer 1-3 sentences), "keywords":[strings], "hint":string }
Rules:
- ALL questions MUST come strictly from the document content.
- Generate exactly ${numQ} questions.
- Type: ${qType==="mcq"?"ALL mcq":qType==="short"?"ALL short":"mix mcq and short roughly equally"}.
- Difficulty: ${difficulty==="easy"?"all easy":difficulty==="medium"?"all medium":difficulty==="hard"?"all hard":"mix easy, medium, hard evenly"}.
- Questions should test comprehension, inference, and recall—not just surface facts.
- Make distractors for MCQ plausible and educational.`;

    try {
      const raw = await gemini(`Document:\n\n${text.slice(0,9000)}\n\nGenerate ${numQ} quiz questions now.`, sys, 3000);
      clearInterval(iv);
      let qs;
      try { const c = raw.replace(/```json|```/gi,"").trim(); qs = JSON.parse(c); if (!Array.isArray(qs)) throw 0; }
      catch { msg("Could not parse questions. Try again.","❌"); setStage("upload"); return; }
      setQ(qs); setAns({}); setFb({}); setConf({}); setFlagged({}); setSummary("");
      if (timerOn) { setTimer(parseInt(timeLimit)); setTA(true); }
      setStage("quiz");
    } catch (e) {
      clearInterval(iv);
      msg("API error. Please try again.","❌");
      setStage("upload");
    }
  };

  // ── Flag ──
  const toggleFlag = (i) => setFlagged(p => ({ ...p, [i]: !p[i] }));

  // ── Confidence ──
  const setConfidence = (i, val) => setConf(p => ({ ...p, [i]: val }));

  // ── Answered count ──
  const answeredCount = Object.keys(answers).filter(k => {
    const v = answers[k]; return v !== undefined && v !== "";
  }).length;
  const flaggedCount = Object.values(flagged).filter(Boolean).length;

  // ── Evaluate ──
  const evaluateAll = async () => {
    clearTimeout(timerRef.current); setTA(false);
    setStage("eval-loading"); setEvalPct(0);

    const fb = {};
    const mcqQs   = questions.filter(q => q.type === "mcq");
    const shortQs = questions.filter(q => q.type === "short");
    const total_steps = 5 + shortQs.length;
    let done = 0;
    const tick = (txt) => { done++; setEvalPct(Math.round((done/total_steps)*100)); setEvalTxt(txt); };

    tick("Parsing all responses…");
    await new Promise(r=>setTimeout(r,400));

    // MCQ - instant scoring
    tick("Evaluating multiple choice questions…");
    mcqQs.forEach(q => {
      const idx = questions.indexOf(q);
      const ua = answers[idx] || "";
      const ok = ua === q.answer;
      fb[idx] = {
        status: ok ? "correct" : "incorrect",
        text: ok ? (q.explanation || "Correct!") : `Correct answer: ${q.answer}. ${q.explanation || ""}`.trim(),
        score: ok ? 10 : 0,
        modelAns: q.answer,
      };
    });
    await new Promise(r=>setTimeout(r,300));

    // Short - AI evaluation
    for (const q of shortQs) {
      const idx = questions.indexOf(q);
      const ua = (answers[idx] || "").trim();
      tick(`Evaluating: "${q.question.slice(0,40)}…"`);
      if (!ua) { fb[idx] = { status:"incorrect", text:"No answer provided.", score:0, modelAns:q.answer }; continue; }
      try {
        const raw = await gemini(
          `Question: ${q.question}\nModel Answer: ${q.answer}\nKey Concepts: ${(q.keywords||[]).join(", ")}\nHint given: ${q.hint||"none"}\nStudent Answer: ${ua}\n\nEvaluate and return ONLY JSON: {"status":"correct"|"partial"|"incorrect","score":0-10,"feedback":"2-3 sentences evaluating accuracy, completeness, and any misconceptions","modelAnswer":"${q.answer}"}`,
          "You are a rigorous but fair academic evaluator. Return ONLY valid JSON, no markdown.", 350
        );
        const p = JSON.parse(raw.replace(/```json|```/gi,"").trim());
        fb[idx] = { status:p.status, text:p.feedback, score:p.score, modelAns:q.answer };
      } catch {
        fb[idx] = { status:"partial", text:"Evaluation failed—reviewed manually.", score:5, modelAns:q.answer };
      }
    }

    setFb(fb);
    tick("Computing topic-wise breakdown…");
    await new Promise(r=>setTimeout(r,300));

    const totPts = Object.values(fb).reduce((s,f)=>s+(f?.score||0),0);
    const maxPts = questions.length * 10;
    const pct = Math.round((totPts/maxPts)*100);

    tick("Generating AI performance summary…");
    const info = questions.map((q,i)=>`Q${i+1}[${q.topic}/${q.difficulty}]:${fb[i]?.status||"?"}`).join(" | ");
    const confInfo = Object.entries(confidence).map(([i,c])=>`Q${+i+1}:${c}`).join(", ");
    try {
      const t = await gemini(
        `Student name: "${userName}". Score: ${pct}% (${totPts}/${maxPts}). Rank: ${getRank(pct).rank}.\nPer-question: ${info}.\nConfidence levels: ${confInfo||"not provided"}.\nWrite 4-5 sentences of honest, personalised, constructive feedback. Address them by name. Highlight what they did well, where they struggled, confidence-accuracy alignment, and actionable next steps.`,
        "You are a supportive academic tutor. Be specific, warm, and encouraging. Write plain text.", 400
      );
      setSummary(t);
    } catch { setSummary(`${userName}, you scored ${pct}%. Review the questions below to deepen your understanding.`); }

    // Save to leaderboard
    const entry = { name: userName, pct, pts: totPts, max: maxPts, rank: getRank(pct).rank, color: getRank(pct).color, time: new Date().toLocaleTimeString() };
    saveLB(entry);
    setLB([...LEADERBOARD]);

    setStage("results");
  };

  // ── Topic breakdown ──
  const topicBreakdown = () => {
    const map = {};
    questions.forEach((q,i) => {
      if (!map[q.topic]) map[q.topic] = { total:0, got:0 };
      map[q.topic].total += 10;
      map[q.topic].got += feedbacks[i]?.score || 0;
    });
    return Object.entries(map).map(([t,{total,got}])=>({ topic:t, pct:Math.round((got/total)*100), got, total }));
  };

  // ── Derived ──
  const totalPts  = Object.values(feedbacks).reduce((s,f)=>s+(f?.score||0),0);
  const maxPts    = questions.length * 10;
  const pct       = maxPts > 0 ? Math.round((totalPts/maxPts)*100) : 0;
  const nCorr     = Object.values(feedbacks).filter(f=>f?.status==="correct").length;
  const nPart     = Object.values(feedbacks).filter(f=>f?.status==="partial").length;
  const nWrong    = Object.values(feedbacks).filter(f=>f?.status==="incorrect").length;
  const rank      = getRank(pct);
  const avgTime   = timerOn && parseInt(timeLimit) > timer ? Math.round((parseInt(timeLimit)-timer)/questions.length) : null;

  // ── Export ──
  const exportResults = () => {
    const lines = [
      `QUIZ RESULTS — ${userName}`,
      `Date: ${new Date().toLocaleString()}`,
      `Score: ${totalPts}/${maxPts} (${pct}%) — Rank: ${rank.rank} (${rank.label})`,
      `Correct: ${nCorr} | Partial: ${nPart} | Incorrect: ${nWrong}`,
      "",
      "=== TOPIC BREAKDOWN ===",
      ...topicBreakdown().map(t=>`${t.topic}: ${t.pct}% (${t.got}/${t.total})`),
      "",
      "=== QUESTION REVIEW ===",
      ...questions.map((q,i) => {
        const fb = feedbacks[i];
        return [
          `Q${i+1} [${q.topic} / ${q.difficulty}] — ${fb?.status||"?"} (${fb?.score||0}/10)`,
          `Question: ${q.question}`,
          `Your Answer: ${answers[i]||"(none)"}`,
          q.type==="mcq"?`Correct Answer: ${q.answer}`:"",
          `Feedback: ${fb?.text||""}`,
          ""
        ].filter(Boolean).join("\n");
      }),
      "",
      "=== AI SUMMARY ===",
      summary,
    ];
    const blob = new Blob([lines.join("\n")], { type:"text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href=url; a.download=`quiz_${userName}_${Date.now()}.txt`; a.click();
    URL.revokeObjectURL(url);
    msg("Results exported!","✅");
  };

  const restart = () => {
    setStage("name"); setUser(""); setDocText(""); setPaste(""); setFile("");
    setQ([]); setAns({}); setFb({}); setConf({}); setFlagged({}); setSummary("");
    setTopics([]); setTA(false); setTimer(0);
  };
  const retake = () => {
    setAns({}); setFb({}); setConf({}); setFlagged({}); setSummary("");
    if (timerOn) { setTimer(parseInt(timeLimit)); setTA(true); }
    setStage("quiz");
  };

  const stageMap = { name:0, upload:1, "gen-loading":2, quiz:2, "eval-loading":3, results:3 };
  const si = stageMap[stage] ?? 0;
  const stageLabels = ["Profile","Document","Quiz","Results"];

  return (
    <>
      <style>{CSS}</style>
      <div className="aurora">
        <div className="aurora-orb ao1"/><div className="aurora-orb ao2"/><div className="aurora-orb ao3"/>
      </div>
      <div className="noise"/>
      <div className="page">

        {/* ── Eval overlay ── */}
        {stage === "eval-loading" && (
          <div className="eval-overlay">
            <div className="eval-spinner"/>
            <div className="eval-title">Evaluating your answers</div>
            <div className="eval-sub">AI is grading every response with care…</div>
            <div className="eval-progress">
              <div className="eval-bar-wrap"><div className="eval-bar" style={{width:`${evalPct}%`}}/></div>
              <div className="eval-step-txt">{evalTxt}</div>
            </div>
          </div>
        )}

        {/* ── NAV ── */}
        <nav className="nav">
          <div className="nav-logo">
            <div className="nav-logo-mark">⚡</div>
            <div className="nav-logo-txt">Quiz<em>.AI Pro</em></div>
          </div>
          <div className="nav-chip">Gemini Flash</div>
          <div className="nav-r">
            {userName && <div className="nav-chip" style={{color:"var(--teal)",borderColor:"rgba(45,212,191,0.3)"}}>👤 {userName}</div>}
            <div className="nav-steps">
              {stageLabels.map((l,i)=>(
                <div key={l} className={`ns ${i<si?"done":i===si?"on":""}`}>{l}</div>
              ))}
            </div>
          </div>
        </nav>

        <div className="wrap">

          {/* ── NAME ── */}
          {stage==="name" && (
            <div className="hero a-up">
              <div className="hero-badge"><div className="hb-dot"/>Document-Based AI Quiz System</div>
              <h1 className="hero-h1">
                <span className="w1">Learn smarter with</span>
                <span className="w2">intelligent quizzing</span>
              </h1>
              <p className="hero-p">
                Upload any document — lecture notes, research papers, study guides. Our AI reads it, generates targeted questions, evaluates your answers, and ranks your performance.
              </p>

              <div className="name-glass">
                <label className="ng-lbl" htmlFor="uname">Your name to get started</label>
                <input
                  id="uname" className="ng-input"
                  placeholder="e.g. Arjun Sharma"
                  value={userName}
                  onChange={e=>setUser(e.target.value)}
                  onKeyDown={e=>e.key==="Enter"&&userName.trim()&&setStage("upload")}
                  autoFocus
                />
                <button
                  className="btn btn-prime" style={{width:"100%",justifyContent:"center"}}
                  onClick={()=>setStage("upload")} disabled={!userName.trim()}
                >Begin → </button>
              </div>

              <div className="feature-pills">
                {[
                  {ico:"📄",txt:"Document Ingestion"},
                  {ico:"🧠",txt:"AI Question Gen"},
                  {ico:"✅",txt:"LLM Evaluation"},
                  {ico:"📊",txt:"Topic Breakdown"},
                  {ico:"🏆",txt:"Rank & Score"},
                  {ico:"📤",txt:"Export Results"},
                  {ico:"🏅",txt:"Leaderboard"},
                  {ico:"⏱",txt:"Timed Mode"},
                ].map(f=>(
                  <div key={f.txt} className="fpill"><span className="fpill-ico">{f.ico}</span>{f.txt}</div>
                ))}
              </div>
            </div>
          )}

          {/* ── UPLOAD ── */}
          {stage==="upload" && (
            <div className="a-up">
              <div className="up-greeting">▸ Welcome back, {userName}</div>
              <h2 className="up-title">Upload your document</h2>
              <p className="up-sub">Provide your study material. Supported: any plain text, markdown, CSV, or JSON file — or simply paste your content below.</p>

              {fileName && (
                <div className="file-pill">
                  <div className="fp-ico">📄</div>
                  <div className="fp-info">
                    <div className="fp-name">{fileName}</div>
                    <div className="fp-meta">{(charCount/1024).toFixed(1)} KB · {wordCount.toLocaleString()} words · {sentCount} sentences</div>
                  </div>
                  <button className="fp-rm" onClick={()=>{setDocText("");setFile("");setTopics([])}}>✕</button>
                </div>
              )}

              {activeText.length > 50 && (
                <div className="doc-stats a-up">
                  <div className="ds"><div className="ds-n">{wordCount.toLocaleString()}</div><div className="ds-l">Words</div></div>
                  <div className="ds"><div className="ds-n">{sentCount}</div><div className="ds-l">Sentences</div></div>
                  <div className="ds"><div className="ds-n">{paraCount}</div><div className="ds-l">Paragraphs</div></div>
                  <div className="ds"><div className="ds-n">{(charCount/1024).toFixed(1)}k</div><div className="ds-l">Characters</div></div>
                </div>
              )}

              <div
                className={`dropzone ${drag?"over":""}`}
                onClick={()=>fileRef.current?.click()}
                onDragOver={e=>{e.preventDefault();setDrag(true)}}
                onDragLeave={()=>setDrag(false)}
                onDrop={e=>{e.preventDefault();setDrag(false);handleFile(e.dataTransfer.files[0])}}
              >
                <div className="dz-ico-wrap">{drag?"🎯":"📂"}</div>
                <div className="dz-title">{drag?"Drop it!":"Drop your document here"}</div>
                <div className="dz-sub">or click to browse files</div>
                <div className="dz-types">{[".txt",".md",".csv",".json",".log"].map(f=><span key={f} className="dz-type">{f}</span>)}</div>
              </div>
              <input ref={fileRef} type="file" accept=".txt,.md,.csv,.json,.log" style={{display:"none"}} onChange={e=>handleFile(e.target.files[0])}/>

              <div className="or-line">or paste text directly</div>
              <textarea className="paste-area" placeholder="Paste lecture notes, an article, chapter text, or any study material here…" value={paste} onChange={e=>{setPaste(e.target.value);setTopics([])}} rows={7}/>

              {topics.length > 0 && (
                <div className="topic-preview a-up">
                  <div className="tp-label">⬡ Detected Topics</div>
                  <div className="tp-tags">
                    {topics.map(t=><div key={t} className="tp-tag">◈ {t}</div>)}
                  </div>
                </div>
              )}

              <div className="sec-div"/>

              <h3 style={{fontSize:18,fontWeight:800,marginBottom:16,letterSpacing:"-0.4px"}}>Quiz Configuration</h3>
              <div className="cfg">
                <div>
                  <label className="cfg-lbl">Questions</label>
                  <select className="cfg-sel" value={numQ} onChange={e=>setNumQ(e.target.value)}>
                    {["3","5","7","10","15"].map(n=><option key={n} value={n}>{n} Questions</option>)}
                  </select>
                </div>
                <div>
                  <label className="cfg-lbl">Question Type</label>
                  <select className="cfg-sel" value={qType} onChange={e=>setQType(e.target.value)}>
                    <option value="mixed">Mixed (MCQ + Short)</option>
                    <option value="mcq">Multiple Choice Only</option>
                    <option value="short">Short Answer Only</option>
                  </select>
                </div>
                <div>
                  <label className="cfg-lbl">Difficulty</label>
                  <select className="cfg-sel" value={difficulty} onChange={e=>setDiff(e.target.value)}>
                    <option value="mixed">Mixed Levels</option>
                    <option value="easy">Easy</option>
                    <option value="medium">Medium</option>
                    <option value="hard">Hard</option>
                  </select>
                </div>
                <div>
                  <label className="cfg-lbl">Timer Mode</label>
                  <select className="cfg-sel" value={timerOn?"on":"off"} onChange={e=>setTimerOn(e.target.value==="on")}>
                    <option value="off">No Timer</option>
                    <option value="on">Timed</option>
                  </select>
                </div>
                {timerOn && (
                  <div>
                    <label className="cfg-lbl">Time Limit</label>
                    <select className="cfg-sel" value={timeLimit} onChange={e=>setTimeL(e.target.value)}>
                      <option value="120">2 Minutes</option>
                      <option value="300">5 Minutes</option>
                      <option value="600">10 Minutes</option>
                      <option value="900">15 Minutes</option>
                    </select>
                  </div>
                )}
              </div>

              <div style={{display:"flex",gap:10,marginTop:24,flexWrap:"wrap"}}>
                <button className="btn btn-prime" onClick={generate} disabled={!activeText.trim()}>
                  ⚡ Generate Quiz
                </button>
                {activeText.length > 100 && topics.length === 0 && (
                  <button className="btn btn-ghost btn-sm" onClick={extractTopics}>🔍 Preview Topics</button>
                )}
              </div>
            </div>
          )}

          {/* ── GEN LOADING ── */}
          {stage==="gen-loading" && (
            <div className="load-screen a-up">
              <div className="spinner"/>
              <div className="load-title">Building your quiz</div>
              <div className="load-sub">Gemini is reading, thinking, and crafting questions…</div>
              <div className="load-list">
                {GEN_STEPS.slice(0,genStep+1).map((s,i)=>(
                  <div key={i} className="ll-item" style={{animationDelay:`${i*0.1}s`}}>
                    <div className="ll-dot"/>{s}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── QUIZ ── */}
          {stage==="quiz" && (
            <div className="a-up">
              <div className="quiz-top">
                <div className="qt-left">
                  <div className="qt-greet">▸ Good luck, {userName}!</div>
                  <div className="qt-title">Answer all questions</div>
                  <div className="qt-sub">{questions.length} questions · Scroll and answer at your pace · 🚩 Flag to review later</div>
                </div>
                <div className="qt-right">
                  {timerOn && (
                    <div className={`qt-timer ${timer < 60 ? "urgent":""}`}>
                      ⏱ {formatTime(timer)}
                    </div>
                  )}
                  <div className="qt-counter"><strong>{answeredCount}</strong> / {questions.length} answered</div>
                </div>
              </div>

              {questions.map((q, qi) => {
                const ua = answers[qi];
                const hasAns = ua !== undefined && ua !== "";
                const isFlagged = flagged[qi];
                const conf = confidence[qi];
                const dc = diffColor[q.difficulty] || "var(--text3)";
                return (
                  <div
                    key={qi}
                    className={`q-card ${hasAns?"answered":""} ${isFlagged?"flagged":""}`}
                    id={`q${qi}`}
                    style={{animationDelay:`${qi*0.05}s`}}
                  >
                    <div className="q-head">
                      <div className="q-num">{qi+1}</div>
                      <div className="q-topic">{q.topic}</div>
                      <div className="q-diff" style={{background:`${dc}15`,borderColor:`${dc}30`,color:dc,border:`1px solid ${dc}30`,fontFamily:"'Fira Code',monospace",fontSize:"10px",padding:"3px 10px",borderRadius:"20px",letterSpacing:"0.06em",textTransform:"uppercase"}}>{q.difficulty}</div>
                      <div className="q-type-chip">{q.type==="mcq"?"MCQ":"Short"}</div>
                      {hasAns && !isFlagged && <div style={{marginLeft:"auto",fontFamily:"'Fira Code',monospace",fontSize:"10px",color:"var(--teal)"}}>✓ Answered</div>}
                      <button className={`q-flag-btn ${isFlagged?"active":""}`} onClick={()=>toggleFlag(qi)} title="Flag for review">🚩</button>
                    </div>
                    <div className="q-body">
                      <div className="q-text">{q.question}</div>

                      {q.type==="mcq" && (
                        <div className="opts">
                          {(q.options||[]).map((opt,oi)=>(
                            <button key={oi} className={`opt ${ua===opt?"sel":""}`} onClick={()=>setAns(p=>({...p,[qi]:opt}))}>
                              <span className="opt-k">{LETTERS[oi]}</span>{opt}
                            </button>
                          ))}
                        </div>
                      )}

                      {q.type==="short" && (
                        <>
                          <textarea
                            className="short-ta"
                            placeholder="Write your answer here…"
                            value={answers[qi]||""}
                            onChange={e=>setAns(p=>({...p,[qi]:e.target.value}))}
                            rows={4}
                          />
                          {q.hint && (
                            <div style={{marginTop:8,fontFamily:"'Fira Code',monospace",fontSize:"11px",color:"var(--text3)",display:"flex",alignItems:"center",gap:7}}>
                              <span style={{color:"var(--amber)"}}>💡</span> Hint: {q.hint}
                            </div>
                          )}
                        </>
                      )}

                      {hasAns && (
                        <div className="confidence-row">
                          <span className="conf-lbl">Confidence:</span>
                          {[{v:"high",l:"High 🟢"},{v:"medium",l:"Medium 🟡"},{v:"low",l:"Low 🔴"}].map(c=>(
                            <button key={c.v} className={`conf-btn conf-${c.v} ${conf===c.v?"active":""}`} onClick={()=>setConfidence(qi,c.v)}>{c.l}</button>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}

              <div style={{height:16}}/>
            </div>
          )}

          {/* ── RESULTS ── */}
          {stage==="results" && (
            <div className="a-up">
              <div className="results-hero">
                <div className="rank-orb" style={{color:rank.color}}>
                  <div className="rank-letter" style={{color:rank.color}}>{rank.rank}</div>
                  <div className="rank-pct">{pct}%</div>
                </div>
                <span className="res-emoji">{rank.emoji}</span>
                <div className="res-user">🎓 {userName}</div>
                <div className="res-label" style={{color:rank.color}}>{rank.label}</div>
                <div className="res-pts">{totalPts} / {maxPts} points · Rank <strong>{rank.rank}</strong>{avgTime?` · Avg ${avgTime}s/question`:""}</div>
              </div>

              <div className="stat-grid">
                <div className="stat"><div className="stat-n g">{nCorr}</div><div className="stat-l">Correct</div></div>
                <div className="stat"><div className="stat-n y">{nPart}</div><div className="stat-l">Partial</div></div>
                <div className="stat"><div className="stat-n r">{nWrong}</div><div className="stat-l">Incorrect</div></div>
                <div className="stat"><div className="stat-n p">{pct}%</div><div className="stat-l">Score</div></div>
                <div className="stat"><div className="stat-n b">{flaggedCount}</div><div className="stat-l">Flagged</div></div>
              </div>

              {/* Topic breakdown */}
              <div className="topic-breakdown">
                <div className="tb-title">Topic-wise Performance</div>
                {topicBreakdown().map((t,i)=>(
                  <div key={i} className="tb-row">
                    <div className="tb-name">{t.topic}</div>
                    <div className="tb-bar-wrap">
                      <div className="tb-bar" style={{width:`${t.pct}%`,background:t.pct>=70?"var(--green)":t.pct>=50?"var(--amber)":"var(--red)",animationDuration:`${0.8+i*0.1}s`}}/>
                    </div>
                    <div className="tb-pct">{t.pct}%</div>
                  </div>
                ))}
              </div>

              {/* AI Summary */}
              {summary && (
                <div className="ai-box">
                  <div className="ai-box-lbl">AI Performance Summary</div>
                  <div className="ai-box-txt">{summary}</div>
                </div>
              )}

              {/* Export */}
              <div className="export-bar">
                <div className="eb-label">📤 Save your detailed results report</div>
                <button className="btn btn-ghost btn-sm" onClick={exportResults}>Download .txt Report</button>
              </div>

              {/* Review */}
              <div className="review-section-hd">Detailed Question Review</div>
              {questions.map((q,i)=>{
                const fb = feedbacks[i];
                const ua = answers[i];
                const conf = confidence[i];
                const isFlagged = flagged[i];
                const cls = fb?.status||"";
                return (
                  <div key={i} className={`rev ${cls}`} style={{animationDelay:`${i*0.04}s`}}>
                    <div className="rev-pills">
                      <span style={{fontFamily:"'Fira Code',monospace",fontSize:"10px",padding:"2px 10px",borderRadius:"20px",background:"rgba(45,212,191,0.07)",border:"1px solid rgba(45,212,191,0.15)",color:"var(--teal)",textTransform:"uppercase",letterSpacing:"0.06em"}}>{q.topic}</span>
                      {fb && <span className={`rpill ${fb.status==="correct"?"c":fb.status==="partial"?"p":"i"}`}>{fb.status==="correct"?"✓ Correct":fb.status==="partial"?"◑ Partial":"✗ Incorrect"}</span>}
                      {fb && <span className="rpill s">{fb.score}/10 pts</span>}
                      <span style={{fontFamily:"'Fira Code',monospace",fontSize:"10px",padding:"2px 8px",borderRadius:"20px",background:diffColor[q.difficulty]+"15",color:diffColor[q.difficulty]}}>{q.difficulty}</span>
                      {isFlagged && <span className="rpill f">🚩 Flagged</span>}
                      {conf && <span style={{fontFamily:"'Fira Code',monospace",fontSize:"10px",color:"var(--text3)"}} className="rev-conf">Confidence: {conf}</span>}
                    </div>
                    <div className="rev-q">Q{i+1}. {q.question}</div>
                    {ua && <div className="rev-ans">Your answer: <span>{ua}</span></div>}
                    {q.type==="mcq" && (
                      <div className="rev-opts">
                        {(q.options||[]).map((opt,oi)=>{
                          let cls2="opt";
                          if(opt===q.answer) cls2+=" corr";
                          else if(opt===ua&&opt!==q.answer) cls2+=" wrong";
                          return <button key={oi} className={cls2} disabled style={{cursor:"default",fontSize:13}}><span className="opt-k">{LETTERS[oi]}</span>{opt}</button>
                        })}
                      </div>
                    )}
                    {q.type==="short" && fb?.modelAns && <div className="rev-ans" style={{marginTop:8}}>Model answer: <span>{fb.modelAns}</span></div>}
                    {fb && <div className="rev-fb">{fb.text}</div>}
                  </div>
                );
              })}

              {/* Leaderboard */}
              {lb.length > 0 && (
                <div className="leaderboard">
                  <div className="lb-title">🏅 Session Leaderboard</div>
                  {lb.map((e,i)=>(
                    <div key={i} className="lb-row" style={e.name===userName&&e.pct===pct?{background:"rgba(123,97,255,0.08)",border:"1px solid rgba(123,97,255,0.2)"}:{}}>
                      <div className={`lb-pos ${i===0?"first":i===1?"second":i===2?"third":""}`}>{i===0?"👑":i===1?"🥈":i===2?"🥉":`#${i+1}`}</div>
                      <div className="lb-name">{e.name}</div>
                      <div className="lb-score">{e.pts}/{e.max} ({e.pct}%)</div>
                      <div className="lb-rank" style={{color:e.color,background:e.color+"15"}}>{e.rank}</div>
                    </div>
                  ))}
                </div>
              )}

              <div className="res-acts">
                <button className="btn btn-prime" onClick={restart}>↩ New Quiz</button>
                <button className="btn btn-teal" onClick={retake}>↺ Retake This Quiz</button>
                <button className="btn btn-ghost" onClick={exportResults}>📤 Export</button>
              </div>
            </div>
          )}

        </div>

        {/* ── Submit bar (quiz only) ── */}
        {stage==="quiz" && (
          <div className="sub-bar">
            <div className="sb-progress">
              <div className="sb-prog-label"><strong>{answeredCount}</strong> of {questions.length} answered</div>
              <div className="sb-prog-bar"><div className="sb-prog-inner" style={{width:`${(answeredCount/questions.length)*100}%`}}/></div>
            </div>
            {flaggedCount>0 && <div className="sb-flags">🚩 {flaggedCount} flagged</div>}
            <div className="sb-mid">
              {flaggedCount>0 && (
                <button className="btn btn-ghost btn-sm" onClick={()=>{
                  const fi = Object.entries(flagged).find(([k,v])=>v&&!feedbacks[k]);
                  if(fi) document.getElementById(`q${fi[0]}`)?.scrollIntoView({behavior:"smooth"});
                }}>Jump to flagged</button>
              )}
              <button
                className="btn btn-prime"
                onClick={evaluateAll}
                disabled={answeredCount===0}
              >
                {answeredCount===questions.length ? "Submit & Evaluate →" : `Submit (${answeredCount}/${questions.length})`}
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
