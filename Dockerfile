FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
LABEL authors="skyr"
COPY requirements.txt .
RUN apt-get update && apt-get install -y sudo && \
    sudo apt-get install software-properties-common -y && \
    sudo apt install apt-utils -y && \
    sudo apt install screen -y && \
    sudo DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:deadsnakes/ppa -y && \
    sudo apt-get update -y && \
    sudo DEBIAN_FRONTEND=noninteractive apt install python3.11 -y && \
    sudo DEBIAN_FRONTEND=noninteractive apt install python3.11-dev -y && \
    sudo DEBIAN_FRONTEND=noninteractive apt install python3.11-dbg -y && \
    sudo DEBIAN_FRONTEND=noninteractive apt install python3.11-full -y && \
    sudo apt-get install python3-dev default-libmysqlclient-dev build-essential pkg-config -y && \
    sudo apt install wget -y && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt

CMD ["bash"]