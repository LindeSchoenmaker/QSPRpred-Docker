FROM python:3.10-slim

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libexpat1 \
    libxext6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/

# Install any needed packages specified in requirements.txt
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install gunicorn

COPY app app
COPY models models
COPY migrations migrations
COPY application.py config.py app.db ./

# Copy entrypoint script
COPY entrypoint.sh /usr/src/entrypoint.sh

RUN chmod a+x /usr/src/entrypoint.sh

ENV FLASK_APP application.py

EXPOSE 5000
# # Define the entrypoint script
ENTRYPOINT ["/usr/src/entrypoint.sh"]

