FROM python:3.8
WORKDIR /app

# Install dependencies
COPY /requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy source code
COPY . /app

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]