import wave
import io
import math
import torch
import torchaudio
import csv
import os
import gradio as gr
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
from fastapi import FastAPI, Response
from pydantic import BaseModel
from gradio_interface import build_interface
import base64

model = "Zyphra/Zonos-v0.1-hybrid"
# model = "Zyphra/Zonos-v0.1-transformer"
model = Zonos.from_pretrained(model, device=device)

# wav, sampling_rate = torchaudio.load("df/audio/cloning_samples/paras_voice.wav")
# wav_derek, sampling_rate_derek = torchaudio.load("df/audio/cloning_samples/derek_voice.wav")

# speaker = model.make_speaker_embedding(wav, sampling_rate)
# speaker_derek = model.make_speaker_embedding(wav_derek, sampling_rate_derek)

max_new_tokens = 86 * 30

# columns = {
#     "first_name": {
#         "col": "INPUT_FIRST_NAME",
#         "speaker_noised": True,
#         "dnsmos_ovrl": 4.0,
#         "fmax": 24000,
#         "vq_val": 0.78,
#         "pitch_std": 39,
#         "speaking_rate": 17,
#         "cfg_scale": 2.0,
#         "min_p": 0.15,
#         "seed": 83952001,
#         "emotions": {
#             "happiness": 0.35,
#             "sadness": 0.16,
#             "disgust": 0.07,
#             "fear": 0.05,
#             "surprise": 0.05,
#             "anger": 0.12,
#             "other": 0.05,
#             "neutral": 0.3
#         }
#     },
#     "value": {
#         "col": "RI Total Value",
#         "speaker_noised": True,
#         "dnsmos_ovrl": 4.0,
#         "fmax": 24000,
#         "vq_val": 0.78,
#         "pitch_std": 39,
#         "speaking_rate": 17,
#         "cfg_scale": 2.0,
#         "min_p": 0.15,
#         "seed": 83952001,
#         "emotions": {
#             "happiness": 0.35,
#             "sadness": 0.16,
#             "disgust": 0.07,
#             "fear": 0.05,
#             "surprise": 0.05,
#             "anger": 0.12,
#             "other": 0.05,
#             "neutral": 0.3
#         }
#     },
#     "county": {
#         "col": "County",
#         "speaker_noised": True,
#         "dnsmos_ovrl": 4.0,
#         "fmax": 24000,
#         "vq_val": 0.78,
#         "pitch_std": 39,
#         "speaking_rate": 17,
#         "cfg_scale": 2.0,
#         "min_p": 0.15,
#         "seed": 83952001,
#         "emotions": {
#             "happiness": 0.35,
#             "sadness": 0.16,
#             "disgust": 0.07,
#             "fear": 0.05,
#             "surprise": 0.05,
#             "anger": 0.12,
#             "other": 0.05,
#             "neutral": 0.3
#         }
#     }
# }

class VMRequest(BaseModel):
    text: str
    speaker_noised: bool = True
    dnsmos_ovrl: float = 4.0
    fmax: int = 24000
    vq_val: float = 0.78
    pitch_std: float = 39.0
    speaking_rate: float = 17.0
    cfg_scale: float = 2.0
    min_p: float = 0.15
    seed: int = 83952001
    emotions: dict = {
        "happiness": 0.35,
        "sadness": 0.16,
        "disgust": 0.07,
        "fear": 0.05,
        "surprise": 0.05,
        "anger": 0.12,
        "other": 0.05,
        "neutral": 0.3
    }
    sample: str = None
    cfg_scale: float = 2.0
    top_p: float = 0.9
    min_k: int = 1024
    min_p: float = 1.0
    linear: float = 2.0
    confidence: float = 2.0
    quadratic: float = 2.0

app = FastAPI()

@app.post("/generate-audio")
def generate_audio_endpoint(request: VMRequest):
    if request.sample:
        audio_bytes = base64.b64decode(request.sample)
        
        # Create a BytesIO object and load it with torchaudio
        audio_buffer = io.BytesIO(audio_bytes)
        wav, sampling_rate = torchaudio.load(audio_buffer)
        
        # Create speaker embedding from the provided sample
        speaker = model.make_speaker_embedding(wav, sampling_rate)
    else:
        speaker = None

    # Generate audio using the new speaker embedding
    buffer = generate_audio(
        request.seed, 
        request.text, 
        "en-us", 
        speaker,  # Using the new speaker embedding
        request.emotions, 
        request.vq_val, 
        request.fmax, 
        request.pitch_std, 
        request.speaking_rate, 
        request.dnsmos_ovrl, 
        request.speaker_noised,
        request.cfg_scale,
        request.top_p,
        request.min_k,
        request.min_p,
        request.linear,
        request.confidence,
        request.quadratic
    )
    return Response(content=buffer.getvalue(), media_type="audio/wav")


def generate_audio(seed, text, language, speaker, emotion, vq_val, fmax, pitch_std, speaking_rate, dnsmos_ovrl, speaker_noised, cfg_scale, top_p, top_k, min_p, linear, confidence, quadratic):
    torch.manual_seed(seed)
    vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)
    emotion_tensor = torch.tensor(list(map(float, emotion.values())), device=device)

    cond_dict = make_cond_dict(
        text=text,
        language=language,
        speaker=speaker,
        emotion=emotion_tensor,
        vqscore_8=vq_tensor,
        fmax=fmax,
        pitch_std=pitch_std,
        speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl,
        speaker_noised=speaker_noised,
        device=device,
        unconditional_keys=[]
    )

    conditioning = model.prepare_conditioning(cond_dict)
    codes = model.generate(
        conditioning,
        # audio_prefix_codes=audio_prefix_codes,
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
        batch_size=1,
        sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear, conf=confidence, quad=quadratic),
        # callback=update_progress,
    )

    wavs = model.autoencoder.decode(codes).cpu()
    
    buffer_ = io.BytesIO()
    torchaudio.save(buffer_, wavs[0], model.autoencoder.sampling_rate, format="wav", encoding="PCM_S", bits_per_sample=16)
    buffer_.seek(0)
    return buffer_

app = gr.mount_gradio_app(app, build_interface(), path="/gradio")

# with open('exports/vm_drop_1.csv') as f:
#     reader = csv.DictReader(f)
#     count = 0
#     for line in reader:
#         count += 1

#         if count > 5:
#             break
        
#         for c in columns.keys():
#             col = columns[c]
#             text = line[col["col"]]
#             if c == "value":
#                 # TODO!!!! USE THIS:    Hey Linda, this is Paaris Shaw with Double Fraction Minerals. I'm reaching out because we're making offers on minerals in Borden county and we'd love to make you an offer in the neighborhood of $12,300 for your interest.
#                 v = float(text.replace("$", "").replace(",", ""))
#                 text = f" and we'd love to make you an offer in the neighborhood of ${int(int(math.ceil(int(v) / 100.0)) * 100):,} for your interest."
#             elif c == "county":
#                 text = "I'm reaching out because we're making offers on minerals in " + text + " county"
#             else:
#                 text = f"Hi {text}, this is Paaris Shaw with Double Fraction Minerals." 

#             print(f"Processing owner #{count}: {c} ({text})")

#             torch.manual_seed(col["seed"])
#             vq_tensor = torch.tensor([col["vq_val"]] * 8, device=device).unsqueeze(0)
#             emotion_tensor = torch.tensor(list(map(float, col["emotions"].values())), device=device)

#             cond_dict = make_cond_dict(
#                 text=text,
#                 language="en-us",
#                 speaker=speaker,
#                 emotion=emotion_tensor,
#                 vqscore_8=vq_tensor,
#                 fmax=float(col["fmax"]),
#                 pitch_std=float(col["pitch_std"]),
#                 speaking_rate=float(col["speaking_rate"]),
#                 dnsmos_ovrl=float(col["dnsmos_ovrl"]),
#                 speaker_noised=float(col["speaker_noised"]),
#                 device=device,
#                 unconditional_keys=[]
#             )

#             count_str = str(count)

#             while len(count_str) < 3:
#                 count_str = "0" + count_str

#             conditioning = model.prepare_conditioning(cond_dict)
#             codes = model.generate(conditioning)
#             wavs = model.autoencoder.decode(codes).cpu()
#             print('sampling_rate', model.autoencoder.sampling_rate)
#             torchaudio.save(f"exports/audio/{count_str}_{c}.wav", wavs[0], model.autoencoder.sampling_rate, encoding="PCM_S", bits_per_sample=16)

#         infiles = [
#             f"exports/audio/{count_str}_first_name.wav",
            
#             f"exports/audio/{count_str}_county.wav",
            
#             f"exports/audio/{count_str}_value.wav",
#             "assets/audio/audio_8.wav",
#             "assets/audio/audio_9.wav",
#             "assets/audio/audio_10.wav",
#         ]

#         outfile = f"exports/audio/{count_str}.wav"

#         data= []
#         for infile in infiles:
#             print('infile', infile)
#             w = wave.open(infile, 'rb')
#             data.append( [w.getparams(), w.readframes(w.getnframes())] )
#             w.close()
            
#         output = wave.open(outfile, 'wb')
#         output.setparams(data[0][0])
#         for i in range(len(data)):
#             output.writeframes(data[i][1])
#         output.close()

#         for c in columns:
#             os.remove(f"exports/audio/{count_str}_{c}.wav")
