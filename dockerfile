# cmdstan installation portion of this dockerfile "inspired" by this: https://github.com/storopoli/cmdstanpy-docker/

FROM python:3.12

# env vars
ENV CSVER=2.35.0
ENV CMDSTAN=/opt/cmdstan-$CSVER

WORKDIR /app

COPY /src /app/src
COPY /data /app/data
COPY /templates /app/templates
COPY requirements.txt /app/requirements.txt
# COPY update_cron /app/update_cron


# ================ Installing CMDSTAN ==================
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
# ===================================================



WORKDIR /app/
RUN pip install --upgrade --no-cache-dir -r requirements.txt
# RUN pip install -r requirements.txt
# Update Database 
RUN ["python", "src/database_helper.py", "--type", "update", "--pathtodb", "data"]



# ======== Cron Job stuff ========
# install cron
RUN apt-get install cron

# Add the cron job
COPY update_cron /etc/cron.d/update_cron

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/update_cron

# Apply cron job
RUN crontab /etc/cron.d/update_cron

# Create the log file to be able to run tail
RUN touch /var/log/cron.log
# ===============================

# Start api and run cron
CMD cron && tail -f /var/log/cron.log & fastapi run ./src/api.py --port 80


EXPOSE 80