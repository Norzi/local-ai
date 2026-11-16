# local-ai: local LLMs and image generation on one RTX 3080 (10 GB)

Notes, settings and scripts from running large language models and diffusion models entirely on my own PC: no cloud, no API, full control over the model and my data.

I started with zero knowledge: I knew the names of the big chat assistants and nothing about what is inside them. Everything here I worked out by experiments, by asking questions and by checking answers against each other. There were no ready-made guides for my conditions: my home connection is capped at 6 Mbit/s and Hugging Face is blocked, so even a 7-8 GB model meant leaving a download running overnight. That is also why I learned to understand file formats before downloading anything.

## Hardware

| Part | Spec |
|---|---|
| GPU | NVIDIA RTX 3080, 10 GB VRAM |
| CPU | Intel Core i5-12400F (6 cores) |
| RAM | 32 GB DDR4-3200 |
| OS | Windows, models and tools on a separate drive |

The 10 GB of VRAM is the main constraint, and most of what is below is about working around it.

---

## Part 1: Local LLMs

### How the setup evolved

| Step | Tool / model | What I learned |
|---|---|---|
| 1 | **Ollama** + Qwen3.5-9B | First model on my own PC. The first answer that came from my own GPU was the moment I got hooked |
| 2 | **LM Studio** | Easier to manage, and easier to add web search |
| 3 | Community fine-tune of Qwen3.5-9B | Vision needs a second file: the **mmproj** |
| 4 | LM Studio as the backend, **Open WebUI** as the frontend (runs on my homelab, see [aegislab](https://github.com/Norzi/aegislab)) | Separating the model server (my PC) from the web UI (always-on server) |
| 5 | **Qwen3.6-35B-A3B** (mixture of experts) on **llama.cpp** | LM Studio gave too little control, I needed to decide which layers go where |
| 6 | The same model inside **opencode** | First agentic tasks (the model reads files, edits code, runs steps) |

### Concepts I had to work out

**Weights, VRAM and GGUF.** At first I did not even know what is stored in VRAM. The model weights are, and they have to fit. GGUF is the format for quantized weights; `Q4`, `Q5`, `Q6` and so on say how many bits per weight are kept. My rule of thumb: Q4 is the floor, below that quality drops visibly.

**Vision needs `mmproj`.** A multimodal model is two parts: the language model and a separate projector file (`mmproj`) that turns images into something the model understands. Without it the model cannot see pictures.

**Context and KV cache.** The context window is not free: the KV cache grows with every token and takes a large share of VRAM. Two settings cut that cost: **flash attention** and **KV cache quantization to q8** (`-ctk q8_0 -ctv q8_0`).

**Mixture of experts (MoE) and CPU offload.** Qwen3.6-35B-A3B has 35B parameters in total but only about 3B are active per token. That makes it possible to keep attention and the KV cache on the GPU and push the expert layers to system RAM with `--n-cpu-moe`, without losing most of the speed.

**Web search.** I connected web search to the model so it can look things up and work with current data instead of only what it learned in training.

**Agentic use.** In opencode the 35B MoE model works well enough for real tasks, but for comfort the context should be at least 128k, because agents fill it quickly.

**RAG.** I understand how retrieval-augmented generation works (embeddings, vector search, feeding retrieved chunks into the prompt). I have not built one yet. That is next, so I can be sure I understand it in practice and not only in theory.

### Measured speed

| Model | Context | What runs where | Speed |
|---|---|---|---|
| Qwen3.5-9B | 16k | everything on the GPU | about 33-40 tokens/s |
| Qwen3.6-35B-A3B | 64k / 128k / 256k | attention + KV cache on the GPU, experts of the first 27 / 30 / 34 layers on the CPU | about 20-24 tokens/s, depending on the context size |

### Going further: keeping the hottest experts in VRAM (RustingStrata)

With `--n-cpu-moe`, the experts that live in system RAM are chosen **by layer**, whatever the model is actually doing. That was the ceiling: about 20-25 tokens/s, and only with careful tuning.

Then I found [**RustingStrata**](https://github.com/RustingStudio/RustingStrata), a third-party inference engine for Qwen3.5/3.6 MoE models written in Rust (it compiles its CUDA kernels at run time and has an OpenAI-compatible server). Experts that do not fit in VRAM run on the CPU, and the placement of experts adapts to the routing it sees: the experts the model uses most stay in VRAM, so most tokens find what they need on the GPU instead of waiting for the CPU. I did not write it; I tested it on my hardware and compared it with what I had.

Same model (Qwen3.6-35B-A3B, Q4_K_M), same hardware (10 GB VRAM, 32 GB RAM):

| Setup | Decode | Prefill |
|---|---|---|
| llama.cpp with `--n-cpu-moe` | about 20-25 tokens/s, with effort | |
| RustingStrata | about **120 tokens/s** (119-123) | about **430 tokens/s** at a 512-token prompt, about **2400 tokens/s** at 4096 |
| RustingStrata with MTP (speculative decoding) | about 150-160 tokens/s | not measured yet |

The raw benchmark output is in [`llm/strata-benchmark.md`](llm/strata-benchmark.md). Prefill speed depends on prompt length, which is why two numbers are shown. The MTP figure comes from my notes of the same test and I will add its log. All numbers come from my own machine and will vary with the prompt, the context size and what else uses the GPU.

What I took from it: a number that looked like a hardware limit was really a **placement problem**. In the benchmark about 87% of the experts the model asked for were already in VRAM, and that is what moved the speed from 20 to 120 tokens/s.

### Launch scripts

Four ready scripts are in [`llm/`](llm/) (llama.cpp `llama-server`, OpenAI-compatible API on port 8899):

| Script | Model | Context | `--n-cpu-moe` |
|---|---|---|---|
| `start-qwen3.5-9b-ctx16k.bat` | Qwen3.5-9B | 16k | not needed |
| `start-qwen3.6-35b-a3b-ctx64k.bat` | Qwen3.6-35B-A3B | 64k | 27 |
| `start-qwen3.6-35b-a3b-ctx128k.bat` | Qwen3.6-35B-A3B | 128k | 30 |
| `start-qwen3.6-35b-a3b-ctx256k.bat` | Qwen3.6-35B-A3B | 256k | 34 |

What the flags do:

| Flag | Why |
|---|---|
| `--n-gpu-layers 99` | put all layers on the GPU first |
| `--n-cpu-moe N` | then move the expert weights of the first N layers back to system RAM |
| `--ctx-size` | context window; a bigger one means a bigger KV cache, so more layers have to move to the CPU (27 -> 30 -> 34) |
| `--flash-attn on`, `-ctk q8_0 -ctv q8_0` | cheaper attention and a half-size KV cache |
| `--fit off` | I set the offload by hand instead of letting llama.cpp guess |
| `-np 1` | one slot, so the whole context belongs to a single conversation |
| `--batch-size 2048 --ubatch-size 1024` | prompt processing speed |
| `--threads 6` | one thread per physical core of the i5-12400F |
| `--models-dir`, `--models-max 1` | models are picked by name from a folder and only one stays loaded at a time |
| `--jinja` | use the chat template from the model |

**The hard part: tuning `--n-cpu-moe`.** The right number depends on how much VRAM is free at that moment (browser, other apps). With `--fit off`, a value that is too low overflows VRAM and generation becomes very slow or simply stops, so I sometimes edit the number in place and restart.

---

## Part 2: ComfyUI

### How the setup evolved

I started with ComfyUI Portable and was surprised that one image model is three downloads: the **diffusion model**, the **text encoder** and the **VAE**. My first model was Krea 2 Turbo, in GGUF with a GGUF text encoder, by analogy with local LLMs. It worked, but slowly, and I did not yet understand what the workflow nodes were doing. Over time I learned the pieces one by one.

### What I learned

**LoRA.** Small add-on files that change a model for a specific style or task without replacing it.

**Reading the speed.** In diffusion the console shows **seconds per iteration (s/it)**, not tokens per second. Lower is better, and the number is per sampling step.

**GGUF is great for LLMs but not for diffusion models.** This was the biggest lesson for speed. A diffusion model runs the full network on every step, and with GGUF the weights have to be dequantized again and again, which costs a lot of time even when everything fits in VRAM. On an RTX 3080 the best format turned out to be **INT8**, because it uses CUDA and tensor cores directly. FP8 does not help here: hardware FP8 acceleration starts with newer GPU generations (40-series and above). Even if an INT8 model does not fit fully and part of it sits in system RAM, it is still faster than GGUF.

**Workflow basics.** Seed (reproducibility), steps, and why **turbo** models use a fixed small number of steps: they are distilled to produce a good result in few steps.

**Text encoder as a prompt helper.** The text encoder of these models is a vision-language model, so the same file can also describe an image or improve a prompt. One model does both jobs, nothing extra to load.

**Two-pass generation.** Instead of generating at high resolution straight away:
1. generate at about 1 MP with 8 steps on the turbo model,
2. upscale the latent 1.7x,
3. run the same model again for 4 steps with denoise 0.5.

The result is sharper, faster and much lighter for the GPU than a single pass at the final resolution. This is exactly what my workflow does.

**Editing, not only generating.** Besides text-to-image there are image-to-image models: editing a finished image, editing only a masked area, and outpainting (I use the Qwen-Image 2.1 outpaint workflow by AusBoss for this). For outpainting I also use Florence-2 for automatic captioning.

**Custom nodes** add everything ComfyUI does not do out of the box.

**SageAttention.** I got it running on Windows, which needed some debugging (a Triton compile error fixed by installing the missing Python headers). In my tests it made little difference on the models I use, so it stays installed, but INT8 was the real speed win.

**Keeping models outside the install.** ComfyUI Portable broke once after a `git pull`, so I reinstalled and now point new installs at the existing models with `extra_model_paths.yaml`, so models are never copied twice.

### My workflow

[`comfyui/krea2-turbo-two-pass.json`](comfyui/krea2-turbo-two-pass.json) is a workflow I built myself around Krea 2 Turbo. It has an optional image-to-prompt step, the two-pass generation described above and a before/after slider. Details in [`comfyui/README.md`](comfyui/README.md). I also use and adapted other community workflows (MOSS-Image for realistic photos, Qwen-Image outpaint), which I do not republish here.

### Examples

All images are generated locally on the RTX 3080. Prompts and workflow metadata are stripped from the files.

| | |
|---|---|
| ![Girl with a lantern in a meadow](docs/img/art-lantern-meadow.jpg) | ![Knight writing a letter](docs/img/art-knight-letter.jpg) |
| Krea 2 Turbo, painterly style | Krea 2 Turbo, painterly style |
| ![Knight in flames](docs/img/art-knight-fire.jpg) | ![Anime girl in a shark hoodie](docs/img/art-shark-hoodie-anime.jpg) |
| Krea 2 Turbo, realistic style | Krea 2 Turbo, anime style |

Realistic photo style with MOSS-Image (AI-generated image, not a real person):

![Photo style: girl on a couch in a shark hoodie](docs/img/photo-moss-shark-hoodie.jpg)

### Models I use

Krea 2 Turbo, Z-Image Turbo and Qwen-Image (INT8 versions), MOSS-Image, Flux.1 Dev, Flux Fill and Flux Kontext for editing and inpainting, Flux.2 Klein (GGUF), Wan 2.2 for image-to-video.

---

## What is next

- Test RAG on top of the faster RustingStrata setup: a small pipeline over my own notes.
- Add the MTP benchmark log and a screenshot of a real chat running on RustingStrata.
- Share my launch scripts and workflows for other people with 10 GB cards.

## Repository layout

```
local-ai/
  README.md
  llm/        # llama.cpp launch scripts for 9B and 35B-A3B
  comfyui/    # my workflow (JSON) and its description
  docs/img/   # example images
```
