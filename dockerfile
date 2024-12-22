# cmdstan installation porttion of this dockerfile "inspired" by this: https://github.com/storopoli/cmdstanpy-docker/

FROM python:3.12

# env vars
ENV CSVER=2.35.0
ENV CMDSTAN=/opt/cmdstan-$CSVER


COPY /src /src
COPY /data /data
COPY /templates /templates
COPY requirements.txt requirements.txt

# install openMPI and MPI's mpicxx binary
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl libopenmpi-dev mpi-default-dev



# set workdir for /opt/cmdstan-CSVER
WORKDIR /opt/

# download and extract cmdstan based on CSVER from github
RUN curl -OL https://github.com/stan-dev/cmdstan/releases/download/v$CSVER/cmdstan-$CSVER.tar.gz \
  && tar xzf cmdstan-$CSVER.tar.gz \
  && rm -rf cmdstan-$CSVER.tar.gz

# copy the make/local to CMDSTAN dir
COPY make/local $CMDSTAN/make/local

# build cmdstan using 2 threads
RUN cd cmdstan-$CSVER \
  && make -j2 build examples/bernoulli/bernoulli

WORKDIR /

# RUN pip install --upgrade --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Create Database 
RUN ["python", "src/database_helper.py", "--type", "update", "--pathtodb", "data"]

# Start api
CMD ["fastapi", "run", "./src/api.py", "--port", "80"]



EXPOSE 80