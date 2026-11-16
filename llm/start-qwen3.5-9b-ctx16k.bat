@echo off
REM Qwen3.5-9B, whole model on the GPU, 16k context
REM Dense model: no --n-cpu-moe needed
cd /d F:\AI\llama.cpp

llama-server.exe ^
  --host 0.0.0.0 --port 8899 ^
  --models-dir F:\AI\models --models-max 1 ^
  --ctx-size 16000 ^
  --n-gpu-layers 99 ^
  -np 1 ^
  --fit off --flash-attn on ^
  -ctk q8_0 -ctv q8_0 ^
  --batch-size 2048 --ubatch-size 1024 ^
  --threads 6 --jinja