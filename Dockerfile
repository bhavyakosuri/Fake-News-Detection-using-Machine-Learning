# Python base
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose port
EXPOSE 8000

# Start
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
