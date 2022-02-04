FROM python:3.7.11

ARG GITHUB_ID
ARG GITHUB_PSWD

ENV GIT_URL="https://"$GITHUB_ID:$GITHUB_PSWD"@github.com/archisketch-dev-team/Real-ESRGAN.git"
RUN git clone $GIT_URL

WORKDIR ./Real-ESRGAN

COPY requirements.txt ./

RUN apt-get update
RUN pip install basicsr
RUN pip install --no-cache-dir -r requirements.txt
RUN python setup.py develop
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models

ENTRYPOINT ["python"]

CMD [ "application.py" ]