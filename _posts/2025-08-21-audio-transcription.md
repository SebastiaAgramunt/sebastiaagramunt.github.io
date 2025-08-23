---
title: Audio Transcription using Whisper from OpenAI
author: sebastia
date: 2025-08-21 17:22:00 +0800
categories: [Misc]
tags: [computer science, speech recognition, AI]
pin: true
toc: true
render_with_liquid: false
math: true
---

Recently I found myself with the need to transcribe an entire [YouTube interview](https://www.youtube.com/watch?v=3GL64FIqgtg&t=912s). The prupose of this post is to use AI to transcribe the audio to text and then translate from Spanish to English.

The interview was hosted by people from [Opground.com](https://opground.com/). In this post I should acknoledge [Eduard Teixidó](https://www.linkedin.com/in/eduardteixidoviladrich/) and [Marcel Gonzalbo](https://www.linkedin.com/in/marcelgozalbobaro/) from Opground for the interview. Also I thank [Lambda AI](https://lambda.ai/) for providing free credit to run the inference in the AI models described in the post.


## Introduction

Transcription is the process of converting speech or audio into written text. As an example, in the spanish congress of deputies, there exist the job of stenographer: A person that writes in paper everything that is said in the chamber to later be saved and published officially. These stenographers perform perfectly their job, they transcribe exactly what is said. Technology however, can help us accelreate the transcription of audio that is already recorded so that we can work with the text. 

One of the first audio-to-text systems was [Audrey by Bell Labs](https://www.bbc.com/future/article/20170214-the-machines-that-learned-to-listen) developed in the 50's of last century. The system was able to recognize phonemes, not words or sentences and "the huge machine occupied a six-foot-high relay rack, consumed substantial power and had streams of cables". 

Audrey was a great breakthrough but clearly not feaseable for practical implementations. Luckyly the field of AI has made a huge progress in the last two decades and one of the models that has great perofrmance is [Whisper from OpenAI](https://openai.com/index/whisper/). Whisper is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web. The large model has around 1.55B parameters or 6.2 GB in floating points of 32 bits. Find more details of the model in the paper [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf) and the github repository [openai/whisper](https://github.com/openai/whisper/tree/main?tab=readme-ov-file). This model is complex... so I defer to the reader to use the references provided to understand the architecture and the backgorund. We will use inference in the model in Spanish and according to [the Readme](https://github.com/openai/whisper/tree/main) of the repository, the `large-v3` model has around 4.7 [Word Error Rate](https://en.wikipedia.org/wiki/Word_error_rate) (WER), which is an impressive metric. In this post we will use Whisper `large-v3` model to transcribe the interview.

## Instance in Lambda AI

As mentioned I'm using [Lambda AI](https://lambda.ai/) as cloud service to use a GPU. Yes, tried running the AI model in my Mac and... surprise, I had to cancel it, was taking too long. I'm using a `gpu_1x_a100_sxm4` machine. Once I'm in the machine I run `gpu_info` (a CLI tool I build on another post) to get the characteristigs of the GPU.

```
Detected 1 CUDA Capable Device(s)

Device 0: NVIDIA A100-SXM4-40GB
  PCI Domain/Bus/Device ID: 0/7/0
  Compute capability: 8.0
  Total global memory: 40442.4 MB
  Free memory (current): 40019.6 MB
  Total allocatable memory (current): 40442.4 MB
  Memory clock rate: 1215 MHz
  Memory bus width: 5120 bits
  L2 cache size: 40960 KB
  Max shared memory per block: 48 KB
  Total constant memory: 64 KB
  Warp size: 32
  Max threads per block: 1024
  Max threads per multiprocessor: 2048
  Multiprocessor count: 108
  Max grid dimensions: [2147483647, 65535, 65535]
  Max block dimensions: [1024, 1024, 64]
  Clock rate: 1410 MHz
  Concurrent kernels: Yes
  ECC enabled: Yes
  Integrated device: No
  Can map host memory: Yes
  Compute mode: Default
  Unified addressing: Yes
  Async engines: 3
  Device overlap: Yes
  PCI bus ID: 7
  PCI device ID: 0
```

This is an Ampere 100 GPU with 40GB of memory, a great GPU for our purposes, inference. Now running `lscpu` you can get the information of the CPU of the machine, won't extend here but just mention that it is an `x86_64` architecture model `AMD EPYC 7J13 64-Core Processor` (check specs [here](https://www.cpubenchmark.net/cpu.php?cpu=AMD+EPYC+7J13&id=4300)). Pretty nice machine inedeed!.


## Downloading audio from YouTube

First we need to download the audio, you can do that directly from youtube. Normally the app we will be using to download uses the cookies from your browswer. That makes things hard as remote machines are normally pure command line and don't have a browser. It is more convenient to download the audio in your local machine and then copy it to your remote machine. 

Use the following script, and name it `download_audio.py`:

```python
import argparse
from pathlib import Path
from yt_dlp import YoutubeDL

def download_audio(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
        ],
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    # Try to resolve final .wav
    expected = out_dir / f"{info.get('title','audio')}.wav"
    if expected.exists():
        return expected
    # Fallback: newest wav in folder
    wavs = list(out_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError("No WAV file produced. Check ffmpeg/yt-dlp output.")
    return max(wavs, key=lambda p: p.stat().st_mtime)

def main():
    ap = argparse.ArgumentParser(description="Download YouTube audio as WAV.")
    ap.add_argument("url", help="YouTube URL")
    ap.add_argument("-o", "--outdir", default="outputs/_tmp", help="Output dir (default: outputs/_tmp)")
    args = ap.parse_args()

    out_dir = Path(args.outdir).resolve()
    wav = download_audio(args.url, out_dir)

if __name__ == "__main__":
    main()
```

Now create a virtual environment and install [yt-dlp](https://github.com/yt-dlp/yt-dlp), I'm using python version `3.12`.

```
rm -rf .venv
python -m venv .venv
.venv/bin/python -m pip install -U yt-dlp
```

Run the command with your video URL:

```bash
PYTHONWARNINGS=ignore .venv/bin/python download_audio.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Now find your `*.wav` file in `outputs/_tmp/` from where you ran the script. Will have the same name as the original video, you can change it to `audio.wav` to make it more simple.

## Copy audio to remote machine

Now with the audio we downloaded we need to copy the file to the remote machine with something like:

```bash
scp -i $HOME/.ssh/id_lambda ~/transcription/outputs/_tmp/audio.wav ubuntu@PUBLIC_IP:/home/ubuntu/audio.wav
```

Changing the `PUBLIC_IP` by your public IP provided by the cloud service. It's pretty straightforward to get it from the Lambda instances webpage. Then the `-i` argument is followed by the private key generated to SSH to the remote machine.

## Run Inference on a machine with GPU

Finally the hard part, use Whisper model to run inference. For that, in the remote machine we will create a new python environment with:

```bash
rm -rf .venv
python -m venv .venv
.venv/bin/pip install -U openai-whisper
```

I got the default system python version as `3.10.12` which is a relatively recent version. Then activate the environment and check that the executable `whisper` is installed:

```bash
source .venv/bin/activate
which whisper
```

Finally run inference using the Ampere 100 GPU with the command:


```bash
whisper audio.wav \
  --model large-v3 \
  --language es \
  --task transcribe \
  --device cuda \
  --fp16 True \
  --temperature 0 \
  --beam_size 1 \
  --output_format txt \
  --output_dir large-v3
```

which will create a directory `large-v3` with the contents `audio.txt`. In the interview I get the first 10 lines with

```bash
cat large-v3/audio.txt | head -10
```

as

```
a las historias de las personas que hacen realidad esta evolución tecnológica, los techies.
Y nada de esto sería posible sin el soporte de Upground, el primer reclutador virtual.
Un sistema basado en inteligencia artificial que replica entrevistas virtuales
y con solo una única entrevista con su chatbot, busca, aplica y gestiona
todas las oportunidades del sector tech por ti.
¿Hay algo por lo que aceptarías un nuevo reto profesional?
No sacrifiques tu tiempo libre, que Upground es tu aliado.
Y con esto empezamos el día de hoy. Hola Marcel.
Hola, ¿qué tal Eduard? Buenos días, buen día. ¿Cómo estamos?
Muy bien, aquí estamos. Hoy por la mañana que tenemos un invitado muy interesante
```

## Translate to english

Use Whisper to translate to english, all parameters are the same but the task, which his `translate` this time.

```bash
whisper audio.wav \
  --model large-v3 \
  --language es \
  --task translate \
  --device cuda \
  --fp16 True \
  --temperature 0 \
  --beam_size 1 \
  --output_format txt \
  --output_dir large-v3_en
```

with the first 10 lines

```
to the stories of the people who make this technological evolution a reality, the techies.
And none of this would be possible without the support of UpGround, the first virtual recruiter.
A system based on artificial intelligence that replicates virtual interviews
and with only one interview with its chatbot,
searches, applies and manages all opportunities in the tech sector for you.
Is there something you would accept as a new professional challenge?
Don't waste your free time, because UpGround is your ally.
And with this we begin today. Hello Marcel.
Hello, how are you Eduard? Good morning, how are you?
Very well, here we are. Today in the morning we have a very interesting guest
```

that seems pretty close to the Spanish version. We made it!.

## Conclusions and future analysis

This has been a quick job, a quick translation. It ran just fine, as a matter of fact, I had to go to the english version and modify parts of the text. It was predicting most words correctly but the context was not understandable sometimes. I don't think this model is wrong, obviously, I just didn't have the time to investigate further. Perhaps my audio quality wasn't good?. Maybe I needed to adjust other parameters like temperature when running the inference?. Anyways, it was a fun exercise that has some practicality for me too. If you reached this part, thank you for reading the post!.