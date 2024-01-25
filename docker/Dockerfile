# FROM nvcr.io/nvidia/pytorch:22.12-py3
#
# RUN pip uninstall -y torch
# COPY ./vllm ./vllm
# ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
# RUN pip install -e ./vllm
# RUN pip install typing_extensions==4.5.0 pydantic==1.10.0 transformers==4.35.0 fastapi==0.101.1
# RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash; apt-get install git-lfs;
#
# CMD ["python", "./vllm/vllm/entrypoints/api_server.py"]

FROM ubuntu:22.04 as lfs
RUN apt-get -y update; apt-get -y install curl
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash; apt-get install git-lfs -y;

FROM vllm/vllm-openai:v0.2.7
RUN cp /usr/bin/python3 /usr/bin/python
COPY --from=lfs /usr/bin/git-lfs /usr/bin/git-lfs
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
