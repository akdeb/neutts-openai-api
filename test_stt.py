"""
Test Kyutai STT directly from microphone - matches original example exactly.
Run: python test_stt.py
"""
import json
import queue

import mlx.core as mx
import mlx.nn as nn
import rustymimi
import sentencepiece
import sounddevice as sd
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils

hf_repo = "kyutai/stt-1b-en_fr-candle"

# Download config
lm_config = hf_hub_download(hf_repo, "config.json")
with open(lm_config, "r") as fobj:
    lm_config = json.load(fobj)

mimi_weights = hf_hub_download(hf_repo, lm_config["mimi_name"])
moshi_name = lm_config.get("moshi_name", "model.safetensors")
moshi_weights = hf_hub_download(hf_repo, moshi_name)
tokenizer = hf_hub_download(hf_repo, lm_config["tokenizer_name"])

lm_config = models.LmConfig.from_config_dict(lm_config)
model = models.Lm(lm_config)
model.set_dtype(mx.bfloat16)

if moshi_weights.endswith(".q4.safetensors"):
    nn.quantize(model, bits=4, group_size=32)
elif moshi_weights.endswith(".q8.safetensors"):
    nn.quantize(model, bits=8, group_size=64)

print(f"loading model weights from {moshi_weights}")
model.load_pytorch_weights(moshi_weights, lm_config, strict=True)

print(f"loading the text tokenizer from {tokenizer}")
text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)

print(f"loading the audio tokenizer {mimi_weights}")
generated_codebooks = lm_config.generated_codebooks
other_codebooks = lm_config.other_codebooks
mimi_codebooks = max(generated_codebooks, other_codebooks)
audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)

print("warming up the model")
model.warmup()

gen = models.LmGen(
    model=model,
    max_steps=4096,
    text_sampler=utils.Sampler(top_k=25, temp=0),
    audio_sampler=utils.Sampler(top_k=250, temp=0.8),
    check=False,
)

block_queue = queue.Queue()

def audio_callback(indata, _frames, _time, _status):
    block_queue.put(indata.copy())

print("recording audio from microphone, speak to get your words transcribed")
print("Press Ctrl+C to stop\n")

last_print_was_vad = False
with sd.InputStream(
    channels=1,
    dtype="float32",
    samplerate=24000,
    blocksize=1920,
    callback=audio_callback,
):
    while True:
        block = block_queue.get()
        block = block[None, :, 0]  # Shape: (1, samples)
        other_audio_tokens = audio_tokenizer.encode_step(block[None, 0:1])
        other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[:, :, :other_codebooks]
        
        text_token, vad_heads = gen.step_with_extra_heads(other_audio_tokens[0])
        if vad_heads:
            pr_vad = vad_heads[2][0, 0, 0].item()
            if pr_vad > 0.5 and not last_print_was_vad:
                print(" [end of turn detected]")
                last_print_was_vad = True
        
        text_token = text_token[0].item()
        if text_token not in (0, 3):
            _text = text_tokenizer.id_to_piece(text_token)
            _text = _text.replace("▁", " ")
            print(_text, end="", flush=True)
            last_print_was_vad = False
