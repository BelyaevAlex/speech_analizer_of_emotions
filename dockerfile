FROM python:3.9

WORKDIR .

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y sox
RUN apt-get install -y ffmpeg

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENV PORT 8080

CMD [ "python", "./main.py" ]