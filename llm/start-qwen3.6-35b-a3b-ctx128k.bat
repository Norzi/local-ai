@echo off
REM Qwen3.6-35B-A3B (MoE), 128k context, expert layers of the first 30 blocks on the CPU
REM --n-cpu-moe depends on the VRAM that is free right now; with --fit off a wrong value is very slow or stalls
cd /d F:\AI\llama.cpp

llama-server.exe ^
  --host 0.0.0.0 --port 8899 ^
  --models-dir F:\AI\models --models-max 1 ^
  --ctx-size 128000 ^
  --n-gpu-layers 99 ^
  --n-cpu-moe 30 ^
  -np 1 ^
  --fit off --flash-attn on ^
  -ctk q8_0 -ctv q8_0 ^
  --batch-size 2048 --ubatch-size 1024 ^
  --threads 6 --jinja