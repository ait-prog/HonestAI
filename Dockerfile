FROM cr.yandex/crp2q2b12lka2f8enigt/pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

RUN pip3 install --no-cache-dir \
    transformers>=4.44.2 \
    accelerate>=0.33.0 \
    sentencepiece \
    protobuf \
    safetensors>=0.4.5

RUN pip3 install --no-cache-dir bitsandbytes

RUN pip3 install --no-cache-dir \
    faiss-cpu \
    sentence-transformers

ENV HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

WORKDIR /workspace
COPY . .
ENTRYPOINT ["python3", "solution.py"]
