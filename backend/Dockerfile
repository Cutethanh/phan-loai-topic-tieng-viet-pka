FROM python:3.11-slim

WORKDIR /app

# Cai thu vien truoc de tan dung bo nho dem cua Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Chep ma nguon
COPY src/ ./src/
COPY backend/ ./backend/

# Thu muc models duoc gan vao luc chay bang docker-compose
EXPOSE 8000

CMD ["uvicorn", "backend.main:ung_dung", "--host", "0.0.0.0", "--port", "8000"]
