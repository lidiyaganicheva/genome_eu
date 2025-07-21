FROM bitnami/spark:3.5.0

USER root

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set Spark to use Python 3
ENV PYSPARK_PYTHON=python3

WORKDIR /

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copy the rest of the code
COPY . .

# Run unit tests — fail the build if any test fails
RUN python3 -m unittest discover -s .

# Run Spark job
CMD ["spark-submit", "--master", "local[*]", "app/genome.py"]
