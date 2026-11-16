# RustingStrata benchmark (Qwen3.6-35B-A3B Q4_K_M, RTX 3080 10 GB, 32 GB RAM)

Command (paths shortened, model name simplified):

```
rustingstrata.exe bench --model "<models>\qwen3.6-35b-a3b\Qwen3.6-35B-A3B-Q4_K_M.gguf" --gpu --pp 512,4096 --tg 128 --reps 3
```

Output (user name removed from the calibration path):

```
cuda device 0: NVIDIA GeForce RTX 3080, sm_86
plan: weights+state+scratch 1984 MB (KV 82 MB for 4224 tokens, prefill 543 MB), expert slots 6557 MB, resident experts 3610/10240
expert traffic: 15.6 MB per layer active set (top-8), 581.2 MB per token over 40 MoE layers
pp    512: prefill 433.6 (130.5-592.6) tok/s | tg 128: decode 123.0 (114.2-128.4) tok/s
expert traffic: 581.2 MB/token -> 77.0 GB/s at 126.4 tok/s
pp   4096: prefill 2426.9 (2058.3-2631.3) tok/s | tg 128: decode 119.2 (117.3-120.6) tok/s
expert traffic: 581.2 MB/token -> 72.9 GB/s at 119.6 tok/s
expert hit rate 87.4% (30884 misses), cache swaps 2788
worker busy 3380.9 us/token, worker stalls 0
```

How to read it:

- **resident experts 3610/10240**: only about a third of the 10240 experts (40 layers x 256) fit in VRAM at once.
- **expert hit rate 87.4%**: despite that, most of the experts the model asks for are already in VRAM.
- **pp 512 / pp 4096**: prompt processing (prefill) for prompts of that many tokens. **tg 128**: generation (decode) of 128 tokens.
- The range in brackets is min-max over 3 repetitions; the first prefill run is the slowest (warm-up).
