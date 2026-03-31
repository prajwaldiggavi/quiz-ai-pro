# 🚀 QuizAI Pro

## AI Quiz Application with Document-Based Question Generation & LLM Evaluation

Transform notes, PDFs, and study material into an intelligent quiz platform with automatic question generation, AI-based answer evaluation, scoring, ranking, topic analysis, and personalized feedback.

---

## 🌐 Live Demo

[https://quiz-ai-pro-airowire.vercel.app/](https://quiz-ai-pro-airowire.vercel.app/)

---

# 📌 Problem Statement

Build a quiz application that accepts a user-provided document, automatically extracts important topics, generates topic-based questions, evaluates answers using AI, and provides scores, explanations, recommendations, and performance analytics.

The project supports:

* Document Upload
* Topic Extraction
* Question Generation
* Subjective and Objective Evaluation
* Ranking and Score Calculation
* Personalized Feedback
* Topic-wise Analysis

---

# 🌟 Project Overview

QuizAI Pro is an AI-powered quiz platform that converts uploaded notes and documents into an interactive exam-like experience.

Users can:

* Upload TXT, CSV, and PDF files
* Paste notes directly
* Choose number of questions
* Select difficulty level
* Select MCQ, Short Answer, or Mixed Mode
* Answer all questions in one interface
* Submit once for full AI evaluation
* Get score, percentage, rank, explanation, and recommendations

Unlike traditional quiz systems, QuizAI Pro creates questions directly from the uploaded content using Retrieval-Augmented Generation (RAG).

---

# ✨ Core Features

## 🧠 Intelligent Question Generation

* Automatic topic-based question generation
* Uses only relevant parts of the uploaded document
* Difficulty Levels:

  * Easy
  * Medium
  * Hard
* Question Modes:

  * MCQ Only
  * Short Answer Only
  * Mixed
* User can select 5–20 questions

---

## 📄 Supported Input Types

* TXT Files
* CSV Files
* PDF Files
* Direct Text / Notes Paste

---

# 📚 Advanced RAG Pipeline

```text
Uploaded Document
        ↓
Text Extraction
        ↓
Text Chunking
        ↓
TF-IDF Embeddings
        ↓
Vector Store
        ↓
Similarity Search
        ↓
Relevant Chunks Retrieved
        ↓
Question Generation
        ↓
Quiz Interface
        ↓
Answer Evaluation
```

Technologies Used:

* Text Chunking
* TF-IDF Embeddings
* Semantic Similarity Search
* Vector Retrieval
* Retrieval-Augmented Generation (RAG)

This ensures that the generated questions come only from the uploaded content.

---

# ⚙️ Full Working Pipeline

```text
1. User enters their name
2. User uploads a document or pastes notes
3. Text is extracted from the file
4. Text is split into smaller chunks
5. Important topics are detected
6. Relevant chunks are selected using similarity search
7. AI generates questions from those chunks
8. User answers all questions
9. AI evaluates every answer
10. Score, rank, analytics, and feedback are displayed
```

---

# 🤖 AI Model Evolution

The project went through multiple AI model stages before selecting the final one.

---

## Stage 1: Gemini Flash API

Initially tested models:

* Gemini 2.0 Flash
* Gemini 2.5 Flash

Why Gemini was first used:

* Very fast responses
* Good for early testing
* Easy API integration
* Useful for quickly testing question generation

Problems:

* API quota exceeded frequently
* Internet connection required
* API keys could be exposed on GitHub
* Mentor suggested avoiding frontier API dependency
* Not suitable for fully open-source offline projects

```text
Gemini Flash
     ↓
Good prototype, not suitable for final project
```

---

## Stage 2: TinyLlama via Ollama

```js
const OLLAMA_MODEL = "tinyllama";
```

Why TinyLlama was tested:

* Very small model
* Works on low-end laptops
* No API key required
* Runs fully offline

Problems:

* Weak reasoning
* Repeated similar questions
* Weak long-answer evaluation
* Sometimes returned invalid JSON
* Missed important topics in long documents

```text
TinyLlama
     ↓
Offline but not powerful enough
```

---

## Stage 3: Final Model — Mistral 7B

```js
const OLLAMA_MODEL = "mistral";
```

Why Mistral was selected:

* Better reasoning
* Better topic extraction
* Better long-answer evaluation
* Better JSON formatting
* Better understanding of large documents
* Runs locally without internet

```text
Gemini Flash
      ↓
TinyLlama
      ↓
Mistral 7B (Final)
```

---

# 📐 Model Comparison

| Model      | Parameters | Approx Size | RAM Needed | Quality |
| ---------- | ---------- | ----------- | ---------- | ------- |
| TinyLlama  | 1.1B       | ~1.1 GB     | 2–4 GB     | Low     |
| Phi-3 Mini | 3.8B       | ~2.3 GB     | 4–6 GB     | Medium  |
| Mistral 7B | 7B         | ~4–7 GB     | 8–12 GB    | High    |
| Llama 3 8B | 8B         | ~5–8 GB     | 10–14 GB   | High    |

Mistral gave the best balance between:

* Speed
* Local execution
* Accuracy
* Better long-answer evaluation
* More reliable question generation

---

# 🧩 Why Ollama Was Used

Ollama allows local LLM execution without using cloud APIs.

Advantages:

* No API cost
* No quota limit
* No internet dependency
* Better privacy
* No API key exposure
* Easy switching between models

Current configuration:

```js
const OLLAMA_BASE = "http://localhost:11434";
const OLLAMA_MODEL = "mistral";
```

---

# 📊 Features Included

* Upload and parse documents
* Paste notes directly
* AI-generated quiz questions
* Topic extraction
* MCQ evaluation
* Subjective answer evaluation
* Difficulty selection
* Sticky submit button
* Progress tracking
* Flag question feature
* Retake quiz option
* Topic-wise analytics
* AI-generated recommendations
* Rank system

---

# 📈 Result Dashboard

After quiz submission, the dashboard displays:

* Total Score
* Percentage
* Rank
* Correct / Wrong Count
* Topic-wise Performance
* Difficulty-wise Analysis
* Personalized Feedback
* Recommendation Section
* Answer Explanation

---

# 🏆 Rank System

| Percentage | Rank |
| ---------- | ---- |
| 95–100%    | S+   |
| 90–94%     | S    |
| 80–89%     | A    |
| 70–79%     | B    |
| 60–69%     | C    |
| 40–59%     | D    |
| Below 40%  | F    |

---

# 📊 Analytics Visualizations

The result page contains:

* Topic-wise Bar Graph
* Correct vs Wrong Pie Chart
* Difficulty-wise Score Graph
* Rank Ring / Progress Circle
* Recommendation Panel

---

# 🎨 UI / UX Highlights

* Glassmorphism design
* Dark futuristic theme
* Animated gradients
* Smooth transitions
* Sticky progress bar
* Responsive design
* Mobile-friendly layout

---

# ⚙️ Tech Stack

## Frontend

* React
* Vite
* JavaScript
* CSS
* Recharts

## AI / NLP

* Ollama
* Mistral 7B
* TF-IDF Embeddings
* Vector Similarity Search
* RAG Pipeline

## File Processing

* TXT Parsing
* CSV Parsing
* PDF Parsing

---

# 🏗️ Project Structure

```text
quiz-ai-pro/
│
├── public/
├── src/
│   ├── components/
│   ├── utils/
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
│
├── docs/
├── package.json
├── vite.config.js
├── README.md
└── .env
```

---

# 🔑 Environment Variables

For Gemini testing only:

```env
VITE_GEMINI_API_KEY=your_api_key_here
```

Final project using Ollama + Mistral:

```env
No API key required
```

---

# 🚀 Installation & Setup

```bash
git clone https://github.com/prajwaldiggavi/quiz-ai-pro.git
cd quiz-ai-pro
npm install
```

Install Ollama:

```bash
ollama pull mistral
```

Start Ollama:

```bash
ollama serve
```

Run the frontend:

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

---

# 💻 Example Ollama API Request

```js
fetch("http://localhost:11434/api/generate", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    model: "mistral",
    prompt: "Generate 5 MCQ questions from this content...",
    stream: false
  })
});
```

---

# 🔬 Development Problems & Improvements

## Problem 1: One Question at a Time

Issue:

* Too many requests
* Slow navigation

Solution:

* Switched to full-page exam mode

---

## Problem 2: Keyword-Based Checking

Issue:

* Correct answers written differently were marked wrong

Solution:

* Added semantic evaluation with LLM + embeddings

---

## Problem 3: Gemini API Limits

Issue:

* API quota exceeded
* Internet dependency

Solution:

* Replaced with Ollama

---

## Problem 4: TinyLlama Weakness

Issue:

* Weak reasoning and repeated questions

Solution:

* Replaced with Mistral 7B

---

# 📅 Future Scope

* OCR for image-based notes
* Save quiz history
* User login system
* Cloud database support
* Multi-language support
* Voice-based answers
* Adaptive difficulty
* Leaderboard and achievements

---

# 👨‍💻 Author

Prajwal Diggavi
ISE Department, BLDE College

Built as part of an AI internship project focused on document-based question generation and LLM-based answer evaluation.

---

# ⭐ Support

If you like this project:

* Star the repository
* Fork the repository
* Share your feedback
* Suggest improvements
