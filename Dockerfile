FROM python:3.10

ARG API_USERNAME
ARG API_PASSWORD
ENV API_USERNAME ${API_USERNAME}
ENV API_PASSWORD ${API_PASSWORD}
ENV TIMEOUT=300
# set api as the current work dir
WORKDIR /api

# copy the requirements list
COPY requirements.txt requirements.txt

# install all the requirements and import corpus
RUN pip install --no-cache-dir --upgrade -r requirements.txt && \
    python -m nltk.downloader stopwords

# copy the main code of fastapi
COPY ./app /api/app

# launch the unicorn server to run the api
# If you are running your container behind a TLS Termination Proxy (load balancer) like Nginx or Traefik,
# add the option --proxy-headers, this will tell Uvicorn to trust the headers sent by that proxy telling it
# that the application is running behind HTTPS, etc.
CMD ["uvicorn", "app.main:codification_ape_app",  "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--timeout-graceful-shutdown", "300"]
