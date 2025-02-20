import gradio as gr
import base64
from zyphra import ZyphraClient
import os
from datetime import datetime
import random

# Initialize Zyphra client
client = ZyphraClient(api_key="YOUR_API_KEY")  # Replace with your API key

# Constants from the documentation
LANGUAGES = {
    "English (US)": "en-us",
    "French": "fr-fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Mandarin Chinese": "cmn"
}

AUDIO_FORMATS = {
    "WAV": "audio/wav",
    "MP3": "audio/mp3",
    "WebM": "audio/webm",
    "Ogg": "audio/ogg",
    "MP4/AAC": "audio/mp4"
}

# Create output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def text_to_speech(text, language, speaking_rate, audio_format, voice_sample=None, seed=420):
    try:
        # Handle voice cloning if a sample is provided
        speaker_audio = None
        if voice_sample is not None:
            with open(voice_sample, "rb") as f:  # Changed from voice_sample.name to voice_sample
                speaker_audio = base64.b64encode(f.read()).decode('utf-8')

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = audio_format.split('/')[-1].replace('mpeg', 'mp3')
        filename = f"generated_speech_{timestamp}.{ext}"
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Generate speech
        result_path = client.audio.speech.create(
            text=text,
            language_iso_code=LANGUAGES[language],
            speaking_rate=speaking_rate,
            mime_type=AUDIO_FORMATS[audio_format],
            speaker_audio=speaker_audio,
            output_path=output_path,
            seed=seed  # Use the seed parameter from the slider
        )

        print(f"Speech generated successfully. Output path: {result_path}")  # Debug log
        return result_path

    except Exception as e:
        print(f"Error in text_to_speech: {str(e)}")  # Debug log
        raise gr.Error(f"Speech generation failed: {str(e)}")


# Create Gradio interface
def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Zyphra API Interface")

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text to Speech",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=3
                )
                language = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="English (US)",
                    label="Language"
                )
                speaking_rate = gr.Slider(
                    minimum=5,
                    maximum=35,
                    value=15,
                    label="Speaking Rate"
                )

                # Rearranged seed controls
                seed = gr.Slider(
                    minimum=-1,
                    maximum=2147483647,
                    value=420,
                    step=1,
                    label="Seed (-1 to 2147483647)"
                )
                randomize_seed = gr.Checkbox(
                    label="Randomize seed",
                    value=False
                )

                audio_format = gr.Dropdown(
                    choices=list(AUDIO_FORMATS.keys()),
                    value="MP3",  # Changed default to MP3 for better compatibility
                    label="Audio Format"
                )
                voice_sample = gr.Audio(
                    label="Voice Sample (Optional)",
                    type="filepath"
                )
                submit_btn = gr.Button("Generate Speech")

            with gr.Column():
                output_audio = gr.Audio(label="Generated Speech")
                status_text = gr.Textbox(label="Status", interactive=False)  # Added status display

        def process_speech(text, language, speaking_rate, audio_format, voice_sample, seed, randomize):
            try:
                if randomize:
                    seed = random.randint(-1, 2147483647)
                result = text_to_speech(text, language, speaking_rate, audio_format, voice_sample, seed)
                return [result, f"Speech generated successfully! (Seed: {seed})"]
            except Exception as e:
                return [None, f"Error: {str(e)}"]

        def toggle_seed_slider(randomize):
            return gr.Slider(interactive=not randomize)

        randomize_seed.change(
            fn=toggle_seed_slider,
            inputs=[randomize_seed],
            outputs=[seed]
        )

        submit_btn.click(
            fn=process_speech,
            inputs=[text_input, language, speaking_rate, audio_format, voice_sample, seed, randomize_seed],
            outputs=[output_audio, status_text]
        )

    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(debug=True)  #debug=True to see more information
