FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY llm_web/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire repository
COPY . .

# Expose the port Hugging Face expects
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "llm_web.backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
