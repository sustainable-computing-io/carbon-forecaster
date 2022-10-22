FROM python:3.10-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

#CMD ["python3.10", "server.py"]
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]