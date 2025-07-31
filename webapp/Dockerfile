FROM python:3.13.3
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Add Broadcom (WSS) client certificate to the systems /etc/ssl/certs/ca-certificates.crt file
ADD ./CertEmulationCA.crt /usr/local/share/ca-certificates/CertEmulationCA.crt
RUN update-ca-certificates

# point to the updated certs for python and openssl
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN pip config set global.cert /etc/ssl/certs/ca-certificates.crt

# Copy just requirements so we don't have to re-do pip install every time code changes
# then run pip install
COPY ./requirements.txt /app/requirements.txt
RUN pip install  --no-cache-dir -r /app/requirements.txt

# Install for pipeline
RUN pip install --no-cache-dir kfp==2.10.1
RUN pip install --no-cache-dir setuptools==80.9.0
#RUN pip install --no-cache-dir opencv-python

# Copy the rest of the code....skipping directories that may contain data
COPY ./MOSCARD /app/src/MOSCARD
COPY ./src /app/src
COPY ./webapp /app/src
COPY ./examples /app/src/examples

# Copy everything in one shot so no layer hides another
WORKDIR /app/src
#COPY ["MOSCARD", "src", "webapp", "examples", "./"]

# Make MOSCARD a proper package
RUN touch /app/src/MOSCARD/__init__.py /app/src/MOSCARD/bin/__init__.py
# Add /app/src/MOSCARD to system path
# Expose the code to Python
ENV PYTHONPATH=/app/src
ENV PYTHONPATH=/app/src/MOSCARD/bin:$PYTHONPATH
ENV PYTHONPATH=/app/src/MOSCARD/data:$PYTHONPATH
ENV PYTHONPATH=/app/src/MOSCARD/model:$PYTHONPATH

#RUN touch /app/src/examples/__init__.py

#RUN ls -l /app/src
# set python path and working dir
#ENV PYTHONPATH=/app/src
#WORKDIR /app/src

# run the flask app!
# host must be 0.0.0.0 for cloud run to pick it up.
# CMD ["python"]
ENTRYPOINT ["sh", "-x", "gunicorn.sh"]