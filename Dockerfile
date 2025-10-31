# Any image is allowed, but this paticular will build significantly faster
# It is a complete copy of pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
FROM cr.yandex/crp2q2b12lka2f8enigt/pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# Установка основных зависимостей
RUN pip3 install --no-cache-dir \
    transformers>=4.44.2 \
    accelerate>=0.33.0 \
    sentencepiece \
    protobuf \
    safetensors>=0.4.5

# Установка зависимостей для 4-bit квантизации (bitsandbytes)
RUN pip3 install --no-cache-dir bitsandbytes

# Установка зависимостей для ретривера (FAISS и sentence-transformers)
RUN pip3 install --no-cache-dir \
    faiss-cpu \
    sentence-transformers

# Установка зависимостей для улучшенной кластеризации (опционально, можно закомментировать если не используются)
# RUN pip3 install --no-cache-dir natasha pymorphy2[fast]

# Переменные окружения для офлайн-режима
ENV HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

WORKDIR /workspace
COPY . .
ENTRYPOINT ["python3", "solution.py"]