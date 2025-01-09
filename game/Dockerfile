FROM python:3.12
WORKDIR /game
COPY ./requirements.txt /game/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /game/requirements.txt
COPY ./evaluate.cpp /game/evaluate.cpp
RUN g++ evaluate.cpp -O2 -o evaluate
COPY . /game

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${API_PORT:-8000}"]
