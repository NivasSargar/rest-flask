FROM bitnami/pytorch
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirments.txt
CMD python app.py