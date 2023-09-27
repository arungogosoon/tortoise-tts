import argparse
import os

import torch
import torchaudio

from api import TextToSpeech, MODELS_DIR
from utils.audio import load_voices, load_audio

clip_paths = ["voices/angie/1.wav", "voices/angie/2.wav", "voices/angie/3.wav"]

reference_clips = [load_audio(p, 22050) for p in clip_paths]
tts = TextToSpeech()
# pcm_audio = tts.tts_with_preset("your text here", voice_samples=reference_clips, preset='fast')


# selected_voices = args.voice.split(',')
# for k, selected_voice in enumerate(clip_paths):
#   if '&' in selected_voice:
#     voice_sel = selected_voice.split('&')
#   else:
#     voice_sel = [selected_voice]
#   voice_samples, conditioning_latents = load_voices(voice_sel)

gen, dbg_state = tts.tts_with_preset("Hello. How are you? ", 
                                    #  k=args.candidates, 
                                     voice_samples=reference_clips, 
                                    #  conditioning_latents=conditioning_latents,
                                    preset="fast", 
                                    # use_deterministic_seed=args.seed, 
                                    return_deterministic_state=True, 
                                    # cvvp_amount=args.cvvp_amount
                                    )
if isinstance(gen, list):
  for j, g in enumerate(gen):
    torchaudio.save(os.path.join("./", 'voice1.wav'), g.squeeze(0).cpu(), 24000)
else:
  torchaudio.save(os.path.join("./", 'voice2.wav'), gen.squeeze(0).cpu(), 24000)
