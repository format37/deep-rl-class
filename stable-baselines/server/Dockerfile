FROM stablebaselines/stable-baselines3
WORKDIR /app
COPY requirements.txt /app
RUN python3 -m pip install -r requirements.txt
COPY train.py /app
CMD ["python3", "train.py"]