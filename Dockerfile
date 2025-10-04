FROM python:3.9-slim# Use an official Python runtime as a parent image

FROM python:3.9-slim

WORKDIR /app

# Set the working directory in the container

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
