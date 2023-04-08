FROM gcr.io/deeplearning-platform-release/tf-cpu.2-11

COPY ["server.py",  "/"]
COPY ["api",  "/api"]
RUN chmod a+x /server.py

RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir thrift
RUN python -m pip install tensorflow_text

WORKDIR /

CMD ["python", "/server.py"]


