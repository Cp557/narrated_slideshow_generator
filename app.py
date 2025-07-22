#!/usr/bin/env python3
"""
Launches a single page Gradio app that:
1. Accepts a topic/question and Gemini API key.
2. Generates slide markdown + TTS for each slide.
3. Lets the user page through slides and hear the narration.
Requires: generate_slideshow.py in the same directory
"""

import asyncio
import atexit
import os
import shutil
import tempfile
import time
import threading
import uuid
from pathlib import Path
from datetime import datetime, timedelta

import gradio as gr
import json
from generate_slideshow import generate_slideshow_with_audio, generate_slideshow_with_audio_async, validate_topic

# Custom CSS for better styling
custom_css = """
.container {max-width: 1000px; margin: auto;}
.input-row {
    margin-bottom: 10px;
}
.api-key-section {
    margin-bottom: 20px;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 8px;
    background-color: #f9f9f9;
}
.api-key-note {
    font-size: 0.9em;
    color: #666;
    margin-top: 5px;
    line-height: 1.5;
}
.demo-row {
    margin-bottom: 20px;
    display: flex;
    flex-direction: column;
}
.demo-row button {
    width: 100% !important;
    height: 40px;
}
.demo-instruction {
    margin-bottom: 5px;
    text-align: center;
    color: #4a6fa5;
}
.demo-instruction h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 500;
}
.slide-container {
    margin: 20px auto;
    padding: 30px;
    border-radius: 10px;
    background-color: white;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    min-height: 300px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.md-container h1, .md-container h2 {
    text-align: center;
    margin-bottom: 20px;
    color: #2c3e50;
}
.slide-image {
    margin: 10px auto;
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
.slide-flex {
    display: flex;
    flex-direction: column;
    gap: 20px;
}
/* Hide image component UI elements */
.slide-image .image-meta,
.slide-image button.icon-button {
    display: none !important;
}
.slide-image > div {
    padding-top: 0 !important;
    border: none !important;
}
.slide-image img {
    border-radius: 8px;
}
.slide-nav {
    display: flex;
    justify-content: space-between;
    margin-top: 15px;
}
.progress-indicator {
    text-align: center;
    font-weight: bold;
    color: #7f8c8d;
}
.app-title {
    text-align: center;
    margin-bottom: 15px;
    background: #7fe4ff;
    color: #222;
    padding: 15px;
    border-radius: 8px;
    border: 4px solid #7fe4ff;
}
"""

# Custom JS for slide transitions
custom_js = """
function animateSlideTransition() {
    const slideContainer = document.querySelector(".slide-container");
    slideContainer.style.opacity = 0;
    slideContainer.style.transform = "translateY(10px)";
    setTimeout(() => { 
        slideContainer.style.opacity = 1; 
        slideContainer.style.transform = "translateY(0px)";
    }, 50);
}
"""

async def generate_presentation_async(topic: str, api_key: str, session_id=None):
    """Async version: Generate slides, audio, and images for all slides; initialise UI with slide 0."""
    topic = (topic or "").strip()
    api_key = (api_key or "").strip()
    
    # Validate API key
    if not api_key:
        raise gr.Error("Please enter your Gemini API key")
        
    # Create or get a session ID
    if session_id is None:
        session_id = get_session_id()
        
    # Initialize session tracking
    if session_id not in active_sessions:
        active_sessions[session_id] = {}
        active_sessions[session_id]["temp_dir"] = tempfile.mkdtemp(prefix=f"gradio_session_{session_id}_")
    
    # Call the async version with the session ID and API key
    slides, audio_files, slide_images = await generate_slideshow_with_audio_async(topic, api_key, session_id=session_id)

    # Basic sanity - keep list lengths aligned for audio
    if len(audio_files) < len(slides):
        audio_files.extend([None] * (len(slides) - len(audio_files)))
    elif len(audio_files) > len(slides):
        audio_files = audio_files[: len(slides)]

    progress_text = f"Slide 1 of {len(slides)}"
    
    initial_image = None
    if slide_images and len(slide_images) > 0:
        initial_image = slide_images[0]

    # Store presentation data in the session
    active_sessions[session_id]["slides"] = slides
    active_sessions[session_id]["audio_files"] = audio_files
    active_sessions[session_id]["slide_images"] = slide_images

    return slides, audio_files, slide_images, 0, slides[0], audio_files[0], initial_image, progress_text, session_id


def generate_presentation(topic: str, api_key: str, session_id=None):
    """Synchronous wrapper for the async presentation generator."""
    # Run the async function and handle empty topic case
    return asyncio.run(generate_presentation_async(topic, api_key, session_id=session_id))


def next_slide(slides, audio, images, idx, session_id):
    global dino_audio_cache, dino_image_cache
    idx = int(idx)
    if idx < len(slides) - 1:
        idx += 1
    progress_text = f"Slide {idx+1} of {len(slides)}"
    
    # Use cached resources if they exist
    audio_path = audio[idx] if idx < len(audio) else None
    image_path = images[idx] if idx < len(images) else None
    
    # Check if audio_path is actually a directory - if so, use None instead
    if audio_path and os.path.isdir(audio_path):
        print(f"Warning: audio_path is a directory: {audio_path}")
        # Create a temp file with the cached audio content if available
        if audio_path in dino_audio_cache:
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.write(dino_audio_cache[audio_path])
            temp_audio.close()
            audio_path = temp_audio.name
        else:
            audio_path = None
    
    # Similarly check the image path
    if image_path and os.path.isdir(image_path):
        print(f"Warning: image_path is a directory: {image_path}")
        # Create a temp file with the cached image content if available
        if image_path in dino_image_cache:
            temp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_img.write(dino_image_cache[image_path])
            temp_img.close()
            image_path = temp_img.name
        else:
            image_path = None
    
    # Use the standard file paths provided (we've already pre-cached them in memory)
    return idx, slides[idx], audio_path, image_path, progress_text


def prev_slide(slides, audio, images, idx, session_id):
    global dino_audio_cache, dino_image_cache
    idx = int(idx)
    if idx > 0:
        idx -= 1
    progress_text = f"Slide {idx+1} of {len(slides)}"
    
    # Use cached resources if they exist
    audio_path = audio[idx] if idx < len(audio) else None
    image_path = images[idx] if idx < len(images) else None
    
    # Check if audio_path is actually a directory - if so, use None instead
    if audio_path and os.path.isdir(audio_path):
        print(f"Warning: audio_path is a directory: {audio_path}")
        # Create a temp file with the cached audio content if available
        if audio_path in dino_audio_cache:
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.write(dino_audio_cache[audio_path])
            temp_audio.close()
            audio_path = temp_audio.name
        else:
            audio_path = None
    
    # Similarly check the image path
    if image_path and os.path.isdir(image_path):
        print(f"Warning: image_path is a directory: {image_path}")
        # Create a temp file with the cached image content if available
        if image_path in dino_image_cache:
            temp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_img.write(dino_image_cache[image_path])
            temp_img.close()
            image_path = temp_img.name
        else:
            image_path = None
    
    # Use the standard file paths provided (we've already pre-cached them in memory)
    return idx, slides[idx], audio_path, image_path, progress_text


def on_close(session_id):
    """Handle cleanup when user closes the browser or refreshes"""
    if session_id:
        cleanup_session(session_id)
    return None


# Set up session management and temporary file handling
active_sessions = {}

def get_session_id():
    """Generate a unique session ID for new user connections"""
    return str(uuid.uuid4())

def cleanup_session(session_id):
    """Remove session data when a user disconnects"""
    if session_id in active_sessions:
        print(f"Cleaning up session {session_id}")
        if "temp_dir" in active_sessions[session_id]:
            temp_dir = active_sessions[session_id]["temp_dir"]
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        active_sessions.pop(session_id, None)

# Register cleanup to happen on exit
def cleanup_all_sessions():
    """Clean up all session data on exit"""
    for session_id in list(active_sessions.keys()):
        cleanup_session(session_id)

def cleanup_old_sessions():
    """Periodically clean up sessions older than 30 minutes"""
    while True:
        try:
            now = time.time()
            # Check all temp dirs in the system temp directory
            temp_root = tempfile.gettempdir()
            for item in os.listdir(temp_root):
                if item.startswith("slideshow_") or item.startswith("gradio_session_"):
                    item_path = os.path.join(temp_root, item)
                    if os.path.isdir(item_path):
                        # Check if directory is older than 30 minutes
                        mtime = os.path.getmtime(item_path)
                        if (now - mtime) > (30 * 60):  # 30 minutes in seconds
                            print(f"Cleaning up old temp directory: {item_path}")
                            shutil.rmtree(item_path, ignore_errors=True)
        except Exception as e:
            print(f"Error in cleanup thread: {e}")
        
        # Sleep for 10 minutes before next cleanup
        time.sleep(600)

# Start the cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True)
cleanup_thread.start()

atexit.register(cleanup_all_sessions)

# Global cache for dinosaur slideshow resources
dino_audio_cache = {}
dino_image_cache = {}

def load_rise_fall_slideshow(api_key):
    """Load cached dinosaur slideshow demo and hide the demo button and instruction."""
    global dino_audio_cache, dino_image_cache
        
    # Clear any previous cache
    dino_audio_cache = {}
    dino_image_cache = {}
    
    cached_dir = Path.cwd()  # Use the current directory instead of a subdirectory
    meta_file = cached_dir / "slides_metadata.json"
    if not meta_file.exists():
        raise gr.Error("Cached slideshow not found.")
    metadata = json.load(open(meta_file))
    slides = []
    for m in metadata:
        md = f"## {m['title']}\n\n" + "\n".join(f"- {b}" for b in m['bullet_points'])
        slides.append(md)
    
    # Get paths but don't load yet
    audio_paths = sorted(str(p) for p in cached_dir.glob("*_slide_*.wav"))
    image_paths = sorted(str(p) for p in cached_dir.glob("*_slide_*_image.jpg"))
    
    # Pre-load all resources in background
    import threading
    def preload_resources():
        for i, img_path in enumerate(image_paths):
            if i < len(image_paths):
                # Cache the image file
                try:
                    with open(img_path, 'rb') as f:
                        dino_image_cache[img_path] = f.read()
                except Exception as e:
                    print(f"Error preloading image {img_path}: {e}")
        
        for i, audio_path in enumerate(audio_paths):
            if i < len(audio_paths):
                # Cache the audio file
                try:
                    with open(audio_path, 'rb') as f:
                        dino_audio_cache[audio_path] = f.read()
                except Exception as e:
                    print(f"Error preloading audio {audio_path}: {e}")
    
    # Start preloading in background
    threading.Thread(target=preload_resources, daemon=True).start()
    
    idx = 0
    progress_text = f"Slide 1 of {len(slides)}"
    
    # Hide the demo button and instruction after clicking
    hide_button = gr.update(visible=False)
    hide_instruction = gr.update(visible=False)
    
    # Process initial audio and image paths to avoid directory errors
    initial_audio = audio_paths[0] if audio_paths else None
    initial_image = image_paths[0] if image_paths else None
    
    # Check if initial audio_path is actually a directory
    if initial_audio and os.path.isdir(initial_audio):
        print(f"Warning: initial audio_path is a directory: {initial_audio}")
        # Create a temp file with the cached audio content if available
        if initial_audio in dino_audio_cache:
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.write(dino_audio_cache[initial_audio])
            temp_audio.close()
            initial_audio = temp_audio.name
        else:
            # Try to find a valid audio file
            wav_files = list(Path(initial_audio).glob("*.wav"))
            if wav_files:
                initial_audio = str(wav_files[0])
            else:
                initial_audio = None
    
    # Similarly check the initial image path
    if initial_image and os.path.isdir(initial_image):
        print(f"Warning: initial image_path is a directory: {initial_image}")
        # Create a temp file with the cached image content if available
        if initial_image in dino_image_cache:
            temp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_img.write(dino_image_cache[initial_image])
            temp_img.close()
            initial_image = temp_img.name
        else:
            # Try to find a valid image file
            jpg_files = list(Path(initial_image).glob("*.jpg"))
            if jpg_files:
                initial_image = str(jpg_files[0])
            else:
                initial_image = None
    
    return slides, audio_paths, image_paths, idx, slides[0], initial_audio, initial_image, progress_text, None, hide_button, hide_instruction

# Gradio theme setup
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    text_size=gr.themes.sizes.text_md,
)

with gr.Blocks(
    title="AI Slideshow Generator", 
    theme=theme,
    css=custom_css,
    js=custom_js,
) as demo:
    gr.Markdown(
        "# Narrated Slideshow Generator 📚💻🗣️",
        elem_classes="app-title"
    )
    
    with gr.Column(elem_classes="container"):
        # API Key section
        with gr.Group(elem_classes="api-key-section"):
            api_key_input = gr.Textbox(
                label="Gemini API Key",
                placeholder="Enter your Gemini API key here",
                type="password",
                scale=1
            )
            gr.HTML(
                """<div class="api-key-note">
                    <strong>Note:</strong> You need a Gemini API key with billing enabled to use this app.<br>
                    Get your API key from <a href="https://aistudio.google.com/app/apikey" target="_blank">Google AI Studio</a>.<br>
                </div>"""
            )
        
        # First row for topic and generate button - increased horizontal width
        with gr.Row(elem_classes="input-row"):
            topic_box = gr.Textbox(
                label="Topic or Question", 
                placeholder="e.g. 'Alexander the Great' or 'How does lightning form?''",
                scale=5  # Increased scale for wider topic box
            )
            gen_btn = gr.Button("Generate", scale=2, variant="primary")
        
        # Second row exclusively for dinosaur button at full width
        with gr.Row(elem_classes="demo-row"):
            demo_instruction = gr.Markdown("### 👇 Click below to view a premade sample slideshow 👇", elem_classes="demo-instruction")
            demo_btn = gr.Button("The Rise and Fall of the Dinosaurs", variant="secondary", scale=1)

        with gr.Group(elem_classes="slide-container"):
            # Create a flex container for slide content and image
            with gr.Column(elem_classes="slide-flex"):
                slide_markdown = gr.Markdown(elem_classes="md-container")
                # Add the title slide image component inside the slide container
                title_image = gr.Image(label="", visible=True, elem_classes="slide-image", show_label=False, show_download_button=False)
            
            progress_indicator = gr.Markdown(
                "Enter a topic and click 'Generate'", 
                elem_classes="progress-indicator"
            )

        with gr.Row(elem_classes="slide-nav"):
            prev_btn = gr.Button("⬅️ Previous Slide", size="sm")
            next_btn = gr.Button("Next Slide ➡️", size="sm", variant="secondary")
        
        audio_player = gr.Audio(autoplay=True, label="Narration", show_label=True)

    # Invisible session state
    slides_state = gr.State([])
    audio_state = gr.State([])
    images_state = gr.State([])
    index_state = gr.State(0)
    session_state = gr.State(None)

    # Wiring
    def prepare_for_generation(topic, api_key, session_id):
        """First step: clear the view and prepare for generation"""
        # First check if the topic is empty
        if not (topic or "").strip():
            gr.Info("Please enter a valid topic or question.")
            return (
                [], [], [], 0, "", None, None, "", session_id,
                gr.update(visible=True),  # Show topic box
                gr.update(visible=True),  # Show generate button
                gr.update(value="Generate", interactive=True),  # Reset button state
                gr.update(visible=True),  # Show dinosaur button
                gr.update(visible=True),  # Show instruction
                False  # should_generate
            )
        
        # Check if API key is provided
        if not (api_key or "").strip():
            gr.Info("Please enter your Gemini API key.")
            return (
                [], [], [], 0, "", None, None, "", session_id,
                gr.update(visible=True),  # Show topic box
                gr.update(visible=True),  # Show generate button
                gr.update(value="Generate", interactive=True),  # Reset button state
                gr.update(visible=True),  # Show dinosaur button
                gr.update(visible=True),  # Show instruction
                False  # should_generate
            )
        
        # Validate the topic using the Gemini Flash input guard
        try:
            if not validate_topic(topic, api_key):
                gr.Info("Please enter a valid topic or question.")
                return (
                    [], [], [], 0, "", None, None, "", session_id,
                    gr.update(visible=True),  # Show topic box
                    gr.update(visible=True),  # Show generate button
                    gr.update(value="Generate", interactive=True),  # Reset button state
                    gr.update(visible=True),  # Show dinosaur button
                    gr.update(visible=True),  # Show instruction
                    False  # should_generate
                )
        except Exception as e:
            gr.Error(f"Error validating topic: {str(e)}. Please check your API key.")
            return (
                [], [], [], 0, "", None, None, "", session_id,
                gr.update(visible=True),  # Show topic box
                gr.update(visible=True),  # Show generate button
                gr.update(value="Generate", interactive=True),  # Reset button state
                gr.update(visible=True),  # Show dinosaur button
                gr.update(visible=True),  # Show instruction
                False  # should_generate
            )
        
        # If topic is valid, clear the current view and prepare for generation
        clear_slide = "Generating your slideshow...\n\nPlease wait."
        gr.Info("This may take a couple minutes.")
        return (
            [], [], [], 0, clear_slide, None, None, "Preparing...", session_id,
            gr.update(visible=False),  # Hide topic box immediately 
            gr.update(visible=False),  # Hide generate button immediately
            gr.update(value="Generating...", interactive=False),  # Update button text and disable
            gr.update(visible=False),  # Hide dinosaur button while generating
            gr.update(visible=False),  # Hide instruction while generating
            True  # should_generate
        )

    def _run_with_new_session(topic, api_key, session_id, should_generate):
        """Second step: actually generate the slideshow"""
        if not should_generate:
            # This case should ideally not be hit if UI updates from prepare_for_generation are correct
            # but as a safeguard, return a state that doesn't proceed.
            return (
                [], [], [], 0, "", None, None, "", session_id,
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(value="Generate", interactive=True),
                gr.update(visible=True),
                gr.update(visible=True),
                False # should_generate (though this output isn't strictly used here, keeping tuple size consistent)
            )
        
        try:
            results = generate_presentation(topic, api_key, session_id)
            return (*results, 
                    gr.update(visible=False), 
                    gr.update(visible=False), 
                    gr.update(value="Generate", interactive=True), 
                    gr.update(visible=False), 
                    gr.update(visible=False),
                    True # should_generate (maintaining tuple size, actual value less critical here)
                   )
        except Exception as e:
            gr.Error(f"Error generating slideshow: {str(e)}")
            return (
                [], [], [], 0, "Error generating slideshow. Please check your API key and try again.", None, None, "", session_id,
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(value="Generate", interactive=True),
                gr.update(visible=True),
                gr.update(visible=True),
                False
            )

    # Two-step process for generation
    # 1. First clear the UI & validate
    should_generate_state = gr.State(False) # Hidden state to pass validation result

    gen_btn.click(
        prepare_for_generation,
        inputs=[topic_box, api_key_input, session_state],
        outputs=[
            slides_state,
            audio_state,
            images_state,
            index_state,
            slide_markdown,
            audio_player,
            title_image,
            progress_indicator,
            session_state,
            topic_box,      # For UI update
            gen_btn,        # For UI update (text, interactivity)
            gen_btn,        # For UI update (visibility - though managed by text/interactivity)
            demo_btn,       # For UI update
            demo_instruction, # For UI update
            should_generate_state # Output of validation
        ]
    ).then(  # 2. Then (conditionally) generate the slideshow
        _run_with_new_session,
        inputs=[topic_box, api_key_input, session_state, should_generate_state],
        outputs=[
            slides_state,
            audio_state,
            images_state,
            index_state,
            slide_markdown,
            audio_player,
            title_image,
            progress_indicator,
            session_state,
            topic_box,      # For UI update post-generation
            gen_btn,        # For UI update post-generation
            gen_btn,        # For UI update post-generation
            demo_btn,       # For UI update post-generation
            demo_instruction, # For UI update post-generation
            should_generate_state # Pass through, though not strictly needed for this output set
        ]
    )
    
    prev_btn.click(
        prev_slide,
        inputs=[slides_state, audio_state, images_state, index_state, session_state],
        outputs=[index_state, slide_markdown, audio_player, title_image, progress_indicator],
    )
    
    next_btn.click(
        next_slide,
        inputs=[slides_state, audio_state, images_state, index_state, session_state],
        outputs=[index_state, slide_markdown, audio_player, title_image, progress_indicator],
    )
    
    # Load cached demo slideshow
    demo_btn.click(
        load_rise_fall_slideshow,
        inputs=[api_key_input],
        outputs=[slides_state, audio_state, images_state, index_state, slide_markdown, audio_player, title_image, 
               progress_indicator, session_state, demo_btn, demo_instruction],
    )
    

if __name__ == "__main__":
    demo.launch()