# 1. Start from a small official Python image
FROM python:3.11-slim

# 2. Work inside /app in the container
WORKDIR /app

# 3. Install libraries first (cached, so rebuilds are fast)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# 4. Copy your app in
COPY app.py .

# 5. Streamlit listens on port 8501
EXPOSE 8501

# 6. The command that runs when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]