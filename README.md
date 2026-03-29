# 🚀 QuizAI Pro

### AI Quiz Application with Document-Based Question Generation & LLM Evaluation

<div align="center">


### ✨ Upload a document → Generate questions → Answer → Get AI evaluation, score, rank & feedback

[🌐 Live Demo – quiz-ai-pro-airowire.vercel.app](https://quiz-ai-pro-airowire.vercel.app/) • [Report](./docs/report.md) • [Research Notes](./docs/research.md)

> 🔗 Deployed on Vercel: [https://quiz-ai-pro-airowire.vercel.app/](https://quiz-ai-pro-airowire.vercel.app/)

</div>

---

# 📌 Problem Statement

Build a quiz application that accepts a user-provided document, automatically generates topic-based questions, and evaluates user answers using an LLM.

The system must support:

* Document ingestion
* Topic extraction
* Question generation
* Answer scoring
* Feedback generation
* Result summary

---

# 🌟 Project Overview

QuizAI Pro is a modern AI-powered quiz platform that transforms uploaded study material into an interactive quiz.

Users can:

* Upload TXT, CSV, or paste notes
* Enter their name and start instantly
* Get automatically generated questions
* Answer all questions in one scrollable exam-like interface
* Submit once for bulk AI evaluation
* Receive marks, grade, rank, feedback, and personalised suggestions

Unlike traditional quiz systems, QuizAI Pro does not require manually writing questions.

---

# 🎯 Key Features

## 🧠 Smart Question Generation

* Generate quizzes directly from uploaded study material
* User can choose:

  * Number of questions
  * Question type:

    * MCQ only
    * Short Answer only
    * Mixed
  * Difficulty level:

    * Easy
    * Medium
    * Hard
* Topic-based question generation from the most relevant parts of the document

## 📄 Supported Inputs

* TXT files
* CSV files
* Pasted notes/text
* Planned: PDF and DOCX

## 📚 Advanced RAG Pipeline

The project uses a complete Retrieval-Augmented Generation pipeline:

```text
Document
   ↓
Text Chunking
   ↓
Embeddings
   ↓
Vector Database
   ↓
Similarity Search
   ↓
Relevant Context
   ↓
Question Generation
```

Technologies used:

* Text chunking
* Embeddings
* Vector search
* RAG
* Semantic similarity

This improves accuracy by generating questions only from the most relevant chunks of the uploaded content.

## ⚙️ Features Included

* Select number of questions
* Select question type
* Select difficulty level
* All questions displayed at once
* Scrollable exam-like interface
* Live progress tracking
* Sticky submit bar
* Retake quiz option

## 🤖 AI Evaluation

* MCQs are evaluated instantly
* Short answers are evaluated together in a single AI request
* Uses semantic comparison and LLM-based scoring
* Generates:

  * Marks
  * Explanation
  * Feedback
  * Suggestions for improvement

## 📊 Result Dashboard

After submission, the system shows:

* Total Score
* Percentage
* Grade
* Rank badge
* Topic-wise performance
* Strengths and weak areas
* Detailed explanation for each answer
* AI-generated summary personalised to the student

## 🏆 Rank Board

| Percentage | Rank |
| ---------- | ---- |
| 95–100%    | S+   |
| 90–94%     | S    |
| 80–89%     | A    |
| 70–79%     | B    |
| 60–69%     | C    |
| 40–59%     | D    |
| Below 40%  | F    |

## 📈 Analytics & Charts

The results page includes:

* Performance charts
* Topic-wise bar graph
* Pie chart for correct vs wrong answers
* Progress graph
* Difficulty-wise comparison
* Rank and score ring visualization

## 🐳 Docker Support

The project is containerized using Docker for easy deployment and reproducibility.

```bash
Docker build → Docker container → Run anywhere
```

---

# 🎨 UI Highlights

* Glassmorphism interface
* Noise texture background
* Smooth animations
* Syne + Outfit + DM Mono typography
* Sticky progress bar
* Live answered-question counter
* Beautiful glowing rank badge
* Responsive design

---

# ⚙️ Tech Stack

## Frontend

* React
* Vite
* CSS / Custom Animations

## Backend / AI

* Ollama
* Mistral 7B or Llama 3
* Optional prototype: Gemini API during early development

## NLP / Retrieval

* Sentence Transformers
* FAISS
* KeyBERT

## Document Processing

* TXT parsing
* CSV parsing
* Planned: PDF support

---

# 🧠 How It Works

```text
1. User enters name
2. Uploads a document
3. Text is extracted
4. Important topics are detected
5. Questions are generated
6. User answers all questions
7. AI evaluates answers
8. Score, rank, and feedback are displayed
```

---

# 📂 Supported File Types

| File Type    | Supported  |
| ------------ | ---------- |
| TXT          | ✅          |
| CSV          | ✅          |
| Pasted Notes | ✅          |
| PDF          | 🚧 Planned |
| DOCX         | 🚧 Planned |

---

# 🏗️ Project Structure

```text
quiz-ai-pro/
│
├── public/
├── src/
│   ├── components/
│   ├── pages/
│   ├── utils/
│   ├── styles/
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
│
├── docs/
│   ├── research.md
│   ├── failed_attempts.md
│   ├── report.md
│   └── screenshots/
│
├── .env
├── package.json
├── vite.config.js
└── README.md
```

---

# 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
VITE_GEMINI_API_KEY=your_api_key_here
```

---

# 🚀 Installation

```bash
git clone https://github.com/prajwaldiggavi/quiz-ai-pro.git
cd quiz-ai-pro
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

---

# 🌍 Deployment

This project can be deployed using:

* Vercel
* Railway
* Netlify

Production build:

```bash
npm run build
```

---

# 🔬 Research Findings

Existing platforms such as Quizizz, Kahoot, and Google Forms require manual question creation.

Limitations found:

* Questions must be created manually
* No automatic document understanding
* Weak subjective answer evaluation
* No detailed AI-generated feedback

QuizAI Pro improves this by:

* Automatically extracting topics
* Generating questions from study material
* Evaluating long answers using AI
* Giving score + explanation + recommendations

---

# ❌ Failed Attempts / Iterations

## Attempt 1: One Question at a Time

Initially, the quiz showed one question at a time.

Problems:

* Too many AI requests
* Slow navigation
* Poor user experience

Improvement:

* Switched to “All Questions at Once” layout

---

## Attempt 2: Keyword-Based Evaluation

Initially, answers were checked only using keyword matching.

Problems:

* Different wording was marked wrong
* Weak subjective answer scoring

Improvement:

* Added semantic evaluation using LLM + embeddings

---

# 📈 Future Improvements

* PDF support
* DOCX support
* OCR for scanned notes
* User login system
* Save quiz history
* Export results as PDF
* Multi-language support
* Voice-based quiz answering
* Difficulty adaptation based on performance

---

# 📅 Daily Development Log

Example commit messages:

```text
Initial repository setup and README
Added TXT and CSV upload support
Implemented topic extraction and question generation
Added all-questions-at-once layout
Implemented bulk AI evaluation and rank system
Improved UI animations and sticky progress bar
Added detailed result summary and recommendations
```

---

# 👨‍💻 Author

**Prajwal**
ISE Department, BLDE College

Built as part of an AI Internship Project focused on document-based quiz generation and LLM evaluation.

---

# ⭐ If You Like This Project

* Star this repository
* Fork the project
* Share feedback

<div align="center">

### Built with AI, curiosity, and modern web technologies ✨

</div>
