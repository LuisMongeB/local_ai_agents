FROM python:3.12.11-bookworm

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

CMD ["fastapi", "run", "--reload"]