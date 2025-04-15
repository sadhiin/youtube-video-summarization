FROM python:3.12-slim

# Set the working directory
WORKDIR /app
# Copy the requirements file into the container
COPY . .
# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000, 8501

# Run the Streamlit app
