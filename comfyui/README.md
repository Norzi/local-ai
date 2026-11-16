# Krea 2 Turbo: two-pass workflow

`krea2-turbo-two-pass.json` is a ComfyUI workflow I built from scratch around Krea 2 Turbo, tuned for an RTX 3080 with 10 GB of VRAM. Drag the file into ComfyUI to load it.

## What it does

```
                    +--> (optional) describe an input image with the text encoder
                    |        |
  your own prompt --+--> [switch] --> positive prompt
                                          |
 Krea 2 Turbo + LoRAs --> pass 1: 8 steps, ~1 MP --> latent upscale 1.7x --> pass 2: 4 steps, denoise 0.5
                                                                                  |
                                                      before / after slider  <---+---> save PNG
```

1. **Prompt switch.** One switch chooses between a prompt I write and a prompt written by the model from an input image.
2. **Image to prompt (optional).** The Qwen3-VL text encoder that Krea 2 already needs for generation is reused to describe an input image, following a fixed instruction (camera, pose, expression, clothing, background, lighting, style). No extra model is loaded.
3. **Pass 1.** 8 steps, CFG 1, `euler` / `simple`, at about 1 megapixel (3:4 portrait by default).
4. **Latent upscale.** x1.7 with bicubic.
5. **Pass 2.** The same model, 4 steps, denoise 0.5: it adds detail instead of redrawing the picture.
6. **Compare.** A slider shows pass 1 against the final image.
7. **Optional depth-control branch** (Krea 2 control LoRA + depth preprocessor), switched off by default.

## Why two passes

Generating straight at the final resolution is slow and heavy for 10 GB of VRAM. A small first pass fixes the composition, and a short second pass at higher resolution finishes the details. It is sharper, faster and lighter for the GPU.

## What you need

| Type | File |
|---|---|
| Diffusion model | Krea 2 Turbo (INT8 is the fastest on RTX 30-series) |
| Text encoder | Qwen3-VL 4B (loader type `krea2`) |
| VAE | `qwen_image_vae.safetensors` |
| LoRAs | any Krea 2 LoRAs, loaded through the Power Lora Loader |
| Custom nodes | `rgthree-comfy` (Power Lora Loader, seed, image comparer), `comfyui_image_metadata_extension` (save with metadata), `comfyui_controlnet_aux` (only for the depth branch) |

Model file names in the JSON are my local names; pick your own files in the loader nodes.

## Note

The example prompt in the workflow is a placeholder. All image paths in the load-image nodes are replaced with `example.png`.
