FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Set environment variables to make installations non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Update and install dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common curl git tzdata \
    build-essential pkg-config python3-dev \
    libcairo2-dev libgirepository1.0-dev

# Configure tzdata non-interactively
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Add deadsnakes repository for different Python versions
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install Python 3.10
RUN apt-get install -y python3.9 python3.9-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip for Python 3.10 using curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools

# Install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

# Cleanup
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Download the base model
RUN python3 -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype='auto', use_safetensors=True, variant='fp16')"

# Download the LoRA weights
RUN mkdir -p /root/.cache/huggingface/hub
RUN git clone https://huggingface.co/Abdullah-Habib/logolora /root/.cache/huggingface/hub/Abdullah-Habib/logolora

# Copy the rest of the application code into the container
COPY . .

# Expose the port on which the application will run
EXPOSE 8080

# Run the FastAPI application using uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]