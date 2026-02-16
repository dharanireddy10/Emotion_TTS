import os
import torch


try:
    from TTS.tts.configs.xtts_config import XttsConfig
    torch.serialization.add_safe_globals([XttsConfig])
except ImportError:
    pass


os.environ["COQUI_TOS_AGREED"] = "1"

import gradio as gr
from TTS.api import TTS


print("Loading XTTS v2 model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

def synthesize_speech(text, reference_audio):
    if not reference_audio:
        raise gr.Error("Please upload a voice sample.")
    
    output_path = "output.wav"
    tts.tts_to_file(
        text=text,
        speaker_wav=reference_audio,
        language="en",
        file_path=output_path
    )
    return output_path

iface = gr.Interface(
    fn=synthesize_speech,
    inputs=[
        gr.Textbox(label="Text to Speak", placeholder="Hello, how are you today?"),
        gr.Audio(type="filepath", label="Reference Voice (Upload a clear sample)")
    ],
    outputs=gr.Audio(label="Generated Voice"),
    title="Emotional Voice Cloning with XTTS"
)

if __name__ == "__main__":
    iface.launch()
