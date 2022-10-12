FROM python:3.6-slim

WORKDIR /home/app/test
COPY . /home/app/test

RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["python","etl_word.py"]
