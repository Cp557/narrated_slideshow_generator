#!/usr/bin/env python3
"""
Generates slide markdown plus TTS audio and images using Gemini models.
Functions exposed:
    generate_slideshow_with_audio(topic, api_key) -> (list_of_slide_markdown, list_of_audio_paths, list_of_image_paths)
"""

import asyncio
import atexit
import datetime
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Optional

from google import genai
from google.genai import types
from PIL import Image

# Deepgram imports for TTS fallback
try:
    from deepgram import DeepgramClient
    # Try different import paths based on SDK version
    try:
        from deepgram.clients.speak.v1.speak_client import SpeakOptions
    except ImportError:
        try:
            from deepgram.clients.speak.v1 import SpeakOptions
        except ImportError:
            from deepgram.clients.speak import SpeakOptions
    DEEPGRAM_AVAILABLE = True
except ImportError:
    print("Deepgram SDK not available. Install with 'pip install deepgram-sdk'")
    DEEPGRAM_AVAILABLE = False

# Remove the global API key - it will be passed as parameter
DEEPGRAM_KEY = os.environ.get("DEEPGRAM_KEY")

# Dictionary to store temporary directories for cleanup
_temp_dirs: Dict[str, str] = {}

def get_temp_dir(session_id: str) -> str:
    """Get or create a temporary directory for a user session"""
    if session_id not in _temp_dirs:
        temp_dir = tempfile.mkdtemp(prefix=f"slideshow_{session_id}_")
        _temp_dirs[session_id] = temp_dir
    return _temp_dirs[session_id]

def cleanup_temp_dirs():
    """Clean up all temporary directories on exit"""
    for session_id, temp_dir in _temp_dirs.items():
        if os.path.exists(temp_dir):
            print(f"Cleaning up temporary directory for session {session_id}")
            shutil.rmtree(temp_dir, ignore_errors=True)
    _temp_dirs.clear()

# Register cleanup function to run on exit
atexit.register(cleanup_temp_dirs)


# ───────────────────────────── Helpers ──────────────────────────────
def _convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Ensure Gemini's raw audio is saved as a proper WAV container."""
    params = _parse_audio_mime_type(mime_type)
    bits_per_sample = params["bits_per_sample"]
    sample_rate = params["rate"]
    num_channels, data_size = 1, len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_data


def _parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    """Extract sample‑rate & bit‑depth from something like audio/L16;rate=24000;…"""
    bits_per_sample, rate = 16, 24_000
    for part in mime_type.split(";"):
        part = part.strip().lower()
        if part.startswith("rate="):
            rate = int(part.split("=", 1)[1])
        elif part.startswith("audio/l"):
            bits_per_sample = int(part.split("l", 1)[1])
    return {"bits_per_sample": bits_per_sample, "rate": rate}


# ────────────────────── JSON Parsing Utilities ───────────────────
import json
import re as _re

def _parse_slides_json(response_text: str) -> list[dict]:
    """Parse the JSON response from Gemini and extract slides data."""
    try:
        # Find JSON content within response if it's not pure JSON
        json_match = re.search(r'```json\s*(.+?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1).strip()
        else:
            json_text = response_text.strip()
            
        # Handle potential JSON formatting issues
        json_text = json_text.replace('\t', ' ')
        
        # Parse the JSON
        slides_data = json.loads(json_text)
        return slides_data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}\nAttempting fallback parsing...")
        return _fallback_parse(response_text)

def _fallback_parse(text: str) -> list[dict]:
    """Fallback method to extract slides if JSON parsing fails."""
    # This is a simple fallback that tries to extract slide content using regex
    slides = []
    
    # Try to find slide content and speaker notes
    content_matches = re.findall(r'"slide_content"\s*:\s*"([^"]+)"', text, re.DOTALL)
    notes_matches = re.findall(r'"speaker_notes"\s*:\s*"([^"]+)"', text, re.DOTALL)
    
    # Create slide entries
    for i in range(min(len(content_matches), len(notes_matches))):
        slides.append({
            "slide_content": content_matches[i].replace('\\n', '\n'),
            "speaker_notes": notes_matches[i]
        })
    
    return slides if slides else _extract_markdown_slides(text)

def _extract_markdown_slides(markdown: str) -> list[dict]:
    """Extract slides from traditional markdown format (for backwards compatibility)."""
    raw = re.split(r"^##\s+", markdown, flags=re.MULTILINE)
    md_slides = [s for s in raw if s.strip()]
    
    result = []
    for slide in md_slides:
        # Preserve title slide format
        if slide.lstrip().startswith("# "):
            content = slide.lstrip()
        else:
            content = f"## {slide}"  # restore header removed by split
            
        # Extract narration if present
        m = re.search(r"Speaker Notes:\s*(.+)", content, flags=re.I | re.S)
        notes = ""
        if m:
            notes = m.group(1).strip()
            # Remove speaker notes from content
            content = content.split("Speaker Notes:")[0].strip()
        
        result.append({"slide_content": content, "speaker_notes": notes})
        
    return result


# ──────────────────────────── Gemini Calls ───────────────────────────
async def _generate_image(prompt: str, output_path: Path, api_key: str) -> str:
    """Generate an image using Gemini Imagen model and save it to the specified path."""
    client = genai.Client(api_key=api_key)

    try:
        # Make this call in a separate thread to not block the event loop
        # since the Gemini client isn't natively async
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: client.models.generate_images(
                model="models/imagen-3.0-generate-002",
                prompt=prompt,
                config=dict(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    person_generation="ALLOW_ADULT",
                    aspect_ratio="16:9",  # Better for slides
                ),
            )
        )

        if not result.generated_images or len(result.generated_images) == 0:
            print("No images generated.")
            return ""

        # Save the generated image
        image = Image.open(BytesIO(result.generated_images[0].image.image_bytes))
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        image.save(output_path)
        return str(output_path)
    except Exception as e:
        print(f"Error generating image: {e}")
        return ""

def _generate_slideshow_markdown(topic: str, api_key: str) -> str:
    """Ask Gemini 2.5 Flash for a markdown deck following strict rules."""
    client = genai.Client(api_key=api_key)
    #model = "gemini-2.5-flash-preview-05-20"
    model = "gemini-2.5-pro-preview-06-05"

    sys_prompt = f"""
<role>
You are SlideGen, an AI that creates fun and engaging narrated slide decks with visual elements about various topics. 
</role>
<instructions>
Create a presentation about '{topic}'. 
Include:
- An introduction slide with bullet points about the overview of the presentation topic and the key areas that will be covered
- 3 content slides with bullet points
- A conclusion slide with bullet points summarizing the key points and insights. 
For each slide provide:
1. Each title should be a single concise and coherent phrase accompanied by exactly one relevant emoji. (Do NOT use the colon ":" format for titles)
2. 3-4 concise bullet points, you will go into more detail in the speaker notes.
3. Clear prose speaker notes suitable for narration that is accessible to general audiences
4. A detailed and specific image prompt for an AI image generator that is relevent to the slide's content. Do not include any text in the image.
Respond with a JSON array where each element represents a slide in the following format:
```json
[
  {{
    "slide_content": "## Introduction Slide Title\n\n",
    "speaker_notes": "Speaker notes",
    "image_prompt": "Image prompt"
  }},
  {{
    "slide_content": "## Content Slide Title\n\n",
    "speaker_notes": "Speaker notes",
    "image_prompt": "Image prompt"
  }},
  {{
    "slide_content": "## Content Slide Title\n\n",
    "speaker_notes": "Speaker notes",
    "image_prompt": "Image prompt"
  }},
  {{
    "slide_content": "## Content Slide Title\n\n",
    "speaker_notes": "Speaker notes",
    "image_prompt": "Image prompt"
  }},
  {{
    "slide_content": "## Conclusion Slide Title\n\n",
    "speaker_notes": "Speaker notes",
    "image_prompt": "Image prompt"
  }},
]
</instructions>
""".strip()

    response = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": sys_prompt}]}],
        config=types.GenerateContentConfig(response_mime_type="text/plain", temperature=0.7),
    )
    return response.text.strip()


async def _generate_tts(narration: str, out_path: Path, api_key: str):
    """GenAI TTS → WAV - Async version with fallback model support"""
    client = genai.Client(api_key=api_key)
    
    # Try with flash model first, then fall back to pro model if needed
    models_to_try = ["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-06-05"]
    
    # Create file with write mode first to ensure it's empty
    with open(out_path, "wb") as _:
        pass
    
    # Try models in sequence until one works
    gemini_exhausted = True
    for model in models_to_try:
        try:
            print(f"Attempting TTS with model: {model}")
            
            stream_instance = client.models.generate_content_stream(
                model=model,
                contents=[{"role": "user", "parts": [{"text": narration}]}],
                config=types.GenerateContentConfig(
                    temperature=1,
                    response_modalities=["audio"] if "tts" in model else [],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Algenib")
                        )
                    ) if "tts" in model else None,
                ),
            )
            
            # Process the stream
            async def process_stream():
                for chunk in stream_instance:
                    if (
                        chunk.candidates
                        and chunk.candidates[0].content
                        and chunk.candidates[0].content.parts
                    ):
                        part = chunk.candidates[0].content.parts[0].inline_data
                        if part and part.data:
                            data = (
                                _convert_to_wav(part.data, part.mime_type)
                                if not part.mime_type.endswith("wav")
                                else part.data
                            )
                            with open(out_path, "ab") as f:
                                f.write(data)
            
            await process_stream()
            # If we get here, the model worked successfully
            print(f"Successfully generated TTS using model: {model}")
            gemini_exhausted = False
            return
                
        except Exception as e:
            if hasattr(e, 'code') and getattr(e, 'code', None) == 429:
                print(f"Model {model} quota exhausted. Trying next model...")
                continue
            else:
                # Re-raise if it's not a quota error
                print(f"Error with model {model}: {e}")
                raise
    
    # If we've tried all Gemini models and none worked, try Deepgram
    if gemini_exhausted and DEEPGRAM_AVAILABLE and DEEPGRAM_KEY:
        try:
            print("Attempting TTS with Deepgram...")
            # Run Deepgram in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: _generate_tts_with_deepgram(narration, out_path))
            print("Successfully generated TTS using Deepgram")
            return
        except Exception as e:
            print(f"Error with Deepgram TTS: {e}")
            # Continue to fallback empty WAV if Deepgram fails
    
    # Last resort fallback - create empty audio file
    print("All TTS models quota exhausted. Creating empty audio file.")
    with open(out_path, "wb") as f:
        f.write(b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x04\x00\x00\x00\x04\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00')


def _generate_tts_with_deepgram(narration: str, out_path: Path):
    """Generate TTS using Deepgram API"""
    # Initialize the Deepgram client
    deepgram = DeepgramClient(DEEPGRAM_KEY)
    print(f"Using Deepgram for TTS generation")
    
    # Configure speech options for v2.x API (which we confirmed works)
    options = SpeakOptions(
        model="aura-2-thalia-en",  # Use Thalia voice
        encoding="linear16",      # This produces WAV format
        container="wav",         # Specify WAV container
        sample_rate=24000        # Sample rate in Hz
    )
    
    # Convert text to speech and save directly to file using the v2.x API
    try:
        response = deepgram.speak.rest.v("1").save(
            str(out_path),          # Output filename
            {"text": narration},    # Text to convert
            options
        )
        print(f"Successfully generated TTS with Deepgram: {out_path}")
        return response
    except Exception as e:
        print(f"Error generating TTS with Deepgram: {e}")
        raise


# ──────────────────────── Public Entry Point ───────────────────
async def generate_slideshow_with_audio_async(topic: str, api_key: str, **kwargs):
    """
    Async version of generate_slideshow_with_audio that processes slides concurrently.
    
    Args:
        topic: The topic to generate a slideshow about
        api_key: Gemini API key
        **kwargs: Optional parameters including session_id
        
    Returns:
        slides_md : list[str]     – markdown for each slide
        audio     : list[str]     – file paths (one per slide, same order)
        images    : list[str|None] – file paths for slide images (one per slide, same order)
    """
    # Get JSON response from Gemini
    json_response = _generate_slideshow_markdown(topic, api_key)
    
    # Parse JSON into slides data
    slides_data = _parse_slides_json(json_response)
    
    # Create temporary directory for this slideshow
    temp_dir = get_temp_dir(str(kwargs.get("session_id", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))))
    safe_topic = re.sub(r"[^\w\s-]", "", topic)[:30]
    safe_topic = re.sub(r"[-\s]+", "-", safe_topic)
    pres_dir = Path(temp_dir) / safe_topic
    pres_dir.mkdir(parents=True, exist_ok=True)

    slides_md = []
    audio_files = []
    slide_images = [None] * len(slides_data)  # Pre-initialize with None values
    
    print("\n====== GENERATING SLIDESHOW CONTENT ======")
    
    # Set up async tasks
    tts_tasks = []
    image_tasks = []
    
    for i, slide_info in enumerate(slides_data, start=1):
        slides_md.append(slide_info["slide_content"])

        # Get the title for logging
        title_match = re.search(r'##\s+(.+?)\n', slide_info["slide_content"].strip())
        title = title_match.group(1) if title_match else f"Slide {i}"
        print(f"\n--- Processing Slide {i}: {title} ---")

        # Generate and print speaker notes
        narration = slide_info.get("speaker_notes", "")
        print("SPEAKER NOTES:")
        print(narration or "No speaker notes provided.")
        
        # Create paths for output files
        wav_path = pres_dir / f"{safe_topic}_slide_{i:02d}.wav"
        audio_files.append(str(wav_path))
        
        # Schedule TTS task
        if narration:
            print(f"Scheduling TTS for slide {i} -> {wav_path}")
            tts_tasks.append(_generate_tts(narration, wav_path, api_key))
        else:
            # Create empty placeholder WAV if no narration
            with open(wav_path, "wb") as f:
                f.write(b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x00\x04\x00\x00\x00\x04\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00')
            print(f"No narration for slide {i}, created empty WAV: {wav_path}")

        # Schedule image generation task
        image_prompt = slide_info.get("image_prompt", "")
        # Append instruction to avoid text in images
        if image_prompt:
            image_prompt = image_prompt.strip() + " Do not include any text in the image."
        print("IMAGE PROMPT:")
        print(image_prompt or "No image prompt provided.")
        if image_prompt:
            image_path = pres_dir / f"{safe_topic}_slide_{i:02d}_image.jpg"
            print(f"Scheduling image for slide {i} -> {image_path}")
            # Store task with index to track which slide it belongs to
            image_tasks.append((i-1, _generate_image(image_prompt, image_path, api_key)))
        else:
            print(f"No image prompt for slide {i}, skipping image generation.")
        
        print("-"*50)
    
    # Execute all TTS tasks concurrently 
    print("\n====== GENERATING TTS IN PARALLEL ======")
    if tts_tasks:
        await asyncio.gather(*tts_tasks)
    print("====== TTS GENERATION COMPLETE ======")
    
    # Execute all image tasks concurrently
    print("\n====== GENERATING IMAGES IN PARALLEL ======")
    if image_tasks:
        # Gather all image generation tasks while preserving their indices
        image_results = await asyncio.gather(*[task for _, task in image_tasks])
        # Map results back to their positions in slide_images
        for (idx, _), path in zip(image_tasks, image_results):
            slide_images[idx] = path
    print("====== IMAGE GENERATION COMPLETE ======")
    
    print("\n====== SLIDESHOW GENERATION COMPLETE ======\n")

    # Ensure all lists have the same length as slides_md
    num_slides = len(slides_md)
    while len(audio_files) < num_slides:
        audio_files.append(None)
    while len(slide_images) < num_slides:
        slide_images.append(None)

    return slides_md, audio_files, slide_images


def generate_slideshow_with_audio(topic: str, api_key: str, **kwargs):
    """
    Synchronous wrapper for the async slideshow generation function.
    Maintains backward compatibility with existing code.
    
    Args:
        topic: The topic to generate a slideshow about
        api_key: Gemini API key
        **kwargs: Optional parameters including:
            - session_id: Unique identifier for the user session
            
    Returns:
        slides_md : list[str]     – markdown for each slide
        audio     : list[str]     – file paths (one per slide, same order)
        images    : list[str|None] – file paths for slide images (one per slide, same order)
    """
    return asyncio.run(generate_slideshow_with_audio_async(topic, api_key, **kwargs))


def validate_topic(topic: str, api_key: str) -> bool:
    """Use Gemini Flash Preview to determine if a topic is suitable for a slideshow."""
    client = genai.Client(api_key=api_key)
    system_prompt = f'''
<role>
You are SlideGenInputGuard, an AI assistant that determines if a user input is a suitable topic for a narrated slideshow presentation.
</role>
<instructions>
Evaluate if "{topic}" is a real-world topic, question, or concept suitable for an educational slideshow. It is fine to include topics that are silly and not real-world topics.
If it is a valid topic, respond with exactly: 1
If it is nonsense, gibberish, meaningless, empty, or not a valid topic, respond with exactly: 0
Only respond with a single digit: 1 or 0. No spaces, newlines or explanations. JUST THE NUMBER 1 OR 0.
</instructions>
<examples>
Input:How does lightning form?
Output:1
Input:The history of horses
Output:1
Input:basketball
Output:1
Input:boobs
Output:1
Input:King Kong
Output:1
Input:Batman
Output:1
Input:Hitler
Output:1
Input:bing bong
Output:0
Input:asdf
Output:0
Input:qwerty
Output:0
Input::)
Output:0
Input:      
Output:0
</examples>
'''.strip()

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[{"role": "user", "parts": [{"text": system_prompt}]}],
        config=types.GenerateContentConfig(response_mime_type="text/plain", temperature=0),
    )
    result = response.text.strip()
    return result == "1"