FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y software-properties-common curl git tzdata \
    build-essential pkg-config python3-dev \
    libcairo2-dev libgirepository1.0-dev \
    cmake

RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

RUN apt-get install -y python3.8 python3.10 python3.10-distutils python3.10-dev

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2 && \
    update-alternatives --set python3 /usr/bin/python3.10

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install pycairo

RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

RUN python3 -c "from diffusers import DiffusionPipeline; import torch; device = 'cuda' if torch.cuda.is_available() else 'cpu'; model = DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16, use_safetensors=True, variant='fp16').to(device)"

RUN mkdir -p /root/.cache/huggingface/hub
RUN git clone https://huggingface.co/Abdullah-Habib/logolora /root/.cache/huggingface/hub/Abdullah-Habib/logolora

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]