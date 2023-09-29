FROM python:3.10-slim-buster
# Install python packages
# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies
RUN pip install -r requirements.txt

# Command to run the application
CMD ["python", "train.py"]