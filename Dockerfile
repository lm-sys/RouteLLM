FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "routellm.openai_server", "--routers", "mf", "--strong-model", "gpt-4-1106-preview", "--weak-model", "anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1"]
