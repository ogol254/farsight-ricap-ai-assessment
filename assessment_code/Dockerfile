FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN addgroup --system app && adduser --system --ingroup app app
COPY requirements.txt .
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && rm -r /wheels
COPY app ./app
COPY assessment_code ./assessment_code
USER app
EXPOSE 8000
# Mount the approved model directory read-only and inject RISK_API_TOKEN at run time.
CMD ["uvicorn", "assessment_code.serve_uc1:app", "--host", "0.0.0.0", "--port", "8000"]
