# qwen3-256k

Qwen3-Embedding-0.6B -> projector -> Qwen3-1.7B pipeline with a ~256k token context on a 24GB GPU (RTX 3090).

Every 512-token block is compressed to its last encoder state. The most recent 2048 tokens stay token-level. The LLM sees about (context / 512 + 2048) positions instead of the full context, so the KV cache stays around 0.3 GB.

The projector and models must be trained with this exact layout first, see `trainStep`. Loss goes on the window part only.

Install:

    pip install torch transformers bitsandbytes

Usage:

    python main.py

Expects a `long_doc.txt` next to the script.
