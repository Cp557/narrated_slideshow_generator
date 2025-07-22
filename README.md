---
title: Narrated Slideshow Generator
sdk: gradio
emoji: 🦀
colorTo: blue
app_file: app.py
tags:
- agent-demo-track
short_description: AI generated slideshows with images and audio.
sdk_version: 5.33.1
pinned: true
license: mit
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/66cf41ae75a88154445c1144/7M0CYMVuVC0Xf1ZqxCyo_.png
---
[Project Overview Video](https://www.canva.com/design/DAGp9q0Pr_I/3XZlutt5Zwhn_tsPN3vZiA/watch?utm_content=DAGp9q0Pr_I&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h0d38006656)

# Narrated Slideshow Generator

AI-powered tool that converts any topic into a narrated slideshow with generated images and audio.

## Install dependencies:
pip install gradio google-genai python-dotenv pillow deepgram-sdk

## Set up API key:
Set up os.environ with your [GEMINI_KEY](https://aistudio.google.com/app/apikey)

## Run:
python app.py

## How it Works
Gemini 2.5 generates slide content and speaker notes

Gemini TTS creates audio narration

Imagen 3 generates slide images

Gradio provides the web interface