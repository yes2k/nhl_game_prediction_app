FROM python:3.12

COPY /src /src
COPY /data /data
COPY /templates /templates
COPY requirements.txt requirements.txt

RUN apt-get update -y
RUN pip install --upgrade --no-cache-dir -r requirements.txt

# Install CmdStan
RUN wget https://github.com/stan-dev/cmdstan/releases/download/v2.36.0/cmdstan-2.36.0.tar.gz && \
    tar -xzf cmdstan-2.36.0.tar.gz && \
    cd cmdstan-2.36.0 && \
    make build -j4 && \
    mv /cmdstan-2.36.0 /opt/cmdstan

# Create Database 
RUN ["python", "src/database_helper.py", "--type", "update", "--pathtodb", "data"]

# Start api
CMD ["fastapi", "run", "./src/api.py", "--port", "80"]



EXPOSE 80