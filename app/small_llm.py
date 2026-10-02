"""Memory-bounded, self-hosted open-weight model for the free CPU deployment.

The small English model is not a validated Somali generator. The caller only
accepts exact source extracts; otherwise it returns retrieved evidence unchanged.
"""
import os
from pathlib import Path
import threading


class SmallLLM:
    name = "SmolLM2-135M-Instruct-Q4_K_M"

    def __init__(self, path=None):
        from llama_cpp import Llama
        path = path or os.getenv("GGUF_PATH", "models/SmolLM2-135M-Instruct-Q4_K_M.gguf")
        if not Path(path).is_file():
            raise RuntimeError("Local GGUF model file is missing")
        self.lock = threading.Lock()
        self.model = Llama(model_path=path, n_ctx=512, n_batch=32, n_threads=1, n_gpu_layers=0, verbose=False)

    def extract(self, question, evidence):
        prompt = f"<|im_start|>system\nCopy the sentence from the evidence that answers the question. Do not add any other information.<|im_end|>\n<|im_start|>user\nEvidence: {evidence}\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"
        with self.lock:
            if len(self.model.tokenize(prompt.encode("utf-8"))) > 400:
                return ""  # Caller returns the source extract; no context overflow.
            result = self.model(prompt, max_tokens=100, temperature=0, stop=["<|im_end|>"])
        return result["choices"][0]["text"].strip().strip('"')
