# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything into the image
COPY . .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Run the app
CMD ["python", "durian_classifier_app.py"]
