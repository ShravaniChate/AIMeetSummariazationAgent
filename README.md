# AI Meeting Assistant

An AI-powered meeting assistant that turns audio into transcripts, summaries, action items, and a chat interface. It accepts YouTube URLs or uploaded audio files and supports Hindi and English meetings.

## Features

- **Audio input**: YouTube URLs or uploaded audio/video files
- **Transcription**: local OpenAI Whisper, with Sarvam AI for Hindi-to-English transcription and translation
- **Summarization**: LangChain (LCEL) pipelines with Google Gemini generate a summary, action items, and key decisions
- **Chat with transcript (RAG)**: ChromaDB and HuggingFace embeddings (`all-MiniLM-L6-v2`) retrieve relevant passages, and Gemini writes the answer
- **Export**: download results as PDF or TXT
- **UI**: Streamlit web app

## Tech Stack

Python, Whisper, Sarvam AI, LangChain (LCEL), Google Gemini, ChromaDB, HuggingFace sentence-transformers, Streamlit

## How It Works

1. Audio is loaded from a YouTube URL or an uploaded file.
2. Whisper transcribes it locally; Sarvam AI handles Hindi-to-English.
3. The transcript is summarized by Gemini through an LCEL chain.
4. The transcript is split into overlapping chunks, embedded, and stored in ChromaDB.
5. Chat questions retrieve the 4 most relevant chunks, which Gemini uses to answer.

## Project Structure
