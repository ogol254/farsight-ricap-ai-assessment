FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app
COPY requirements.txt requirements-connected.txt requirements-neural.txt ./
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-neural.txt
COPY --chown=appuser:appuser . .
USER appuser
ENV HF_HOME=/home/appuser/.cache/huggingface ENABLE_NEURAL=1 PYTHONUNBUFFERED=1
RUN python -c "from app.models import Models; Models(); from app.ocr import MeterOCR; MeterOCR()"
RUN python -c "from sentence_transformers import SentenceTransformer; from transformers import AutoTokenizer,AutoModelForCausalLM; SentenceTransformer('intfloat/multilingual-e5-small'); AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"
EXPOSE 7860
CMD ["uvicorn", "app.connected:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--no-access-log"]
