import torchaudio
import torch

from models import Model
from generator import Generator

CACHE_DIR = None  # /path/to/cache eg /home/user/storage/csm1b

def load_csm_1b(pth: str = "KandirResearch/csm-1b-safetensors", device: str = "cuda") -> Generator:
    model = Model.from_pretrained(pth, cache_dir=CACHE_DIR)
    model.to(device=device, dtype=torch.bfloat16)

    generator = Generator(model)
    return generator

generator = load_csm_1b()
with torch.inference_mode():
    audio = generator.generate(
        text="hello bbepepep beep boop i'm a computer",
        speaker=0,
        context=[],
        max_audio_length_ms=50_000,
    )

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
