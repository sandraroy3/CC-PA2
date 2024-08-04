# Use the official Spark base image
FROM ubuntu:20.04

# Set environment variables
ENV SPARK_HOME /opt/spark

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Copy the prediction script
COPY predict_model.py /opt/spark/predict_model.py

# Install dependencies
RUN apt-get update && apt-get install -y python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install pyspark numpy

# Set the entrypoint
ENTRYPOINT ["python3", "/opt/spark/predict_model.py"]