# 🎥 Video to Blog Generator with Whisper & Qwen3

This project converts uploaded videos into blog posts by first transcribing the audio using OpenAI's Whisper model, then generating a natural, human-like blog article using the Qwen/Qwen2.5-7B-Instruct language model.

---

## Features

- **Automatic Speech Recognition**: Extracts transcript from video audio using Whisper-large-v3.
- **Blog Generation**: Creates a readable blog post based strictly on the transcript using Qwen3.
- **Easy to Use**: Simple Gradio web interface for uploading videos and viewing results.
- **Local GPU Acceleration**: Supports CUDA if available for faster inference.

---

## Requirements

- Python 3.10 or later
- CUDA-enabled GPU recommended but not required
- [FFmpeg](https://ffmpeg.org/) installed on your system (required for audio extraction)
- Create a static/uploads folder in the project root

Install dependencies via:

pip install -r requirements.txt
