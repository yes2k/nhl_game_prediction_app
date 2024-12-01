FROM python:3.12

COPY /src /src
COPY /data /data
COPY /templates /templates
COPY requirements.txt requirements.txt

RUN pip install --upgrade --no-cache-dir -r requirements.txt

# Install Cmdstan 
RUN ["python", "-c", "import cmdstanpy; cmdstanpy.install_cmdstan(dir='/opt/cmdstan', version='2.35')"]

# Create Database 
RUN ["python", "src/database_helper.py", "--type", "update", "--pathtodb", "data"]

# Start api
CMD ["fastapi", "run", "./src/api.py", "--port", "80"]



EXPOSE 80