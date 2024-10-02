FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Update and install dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common curl

# Add deadsnakes repository for different Python versions
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install Python 3.10
RUN apt-get install -y python3.10 python3.10-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip for Python 3.10 using curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Cleanup
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app


# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Run the database script

# Expose the port on which the application will run
EXPOSE 8080

# Run the FastAPI application using uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
# uvicorn app:app --host 0.0.0.0 --port 8080