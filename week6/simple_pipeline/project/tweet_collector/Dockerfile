FROM python:3.6-slim

WORKDIR /home/app/
COPY . /home/app/

RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["python","get_tweets.py"]