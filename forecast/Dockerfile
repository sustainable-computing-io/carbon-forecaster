FROM python:3.10-buster

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

#CMD ["python3.10", "server.py"]
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]