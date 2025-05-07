# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Install uv (a fast Python package installer)
RUN pip install uv

# 4. Copy the dependency management files
COPY pyproject.toml uv.lock ./

# 5. Install project dependencies using uv
# Assuming pyproject.toml contains dependencies. If it's for project metadata only,
# and dependencies are listed elsewhere or need to be extracted, this step might need adjustment.
# For now, we'll attempt to install from pyproject.toml directly if uv supports it,
# or rely on uv.lock.
# uv pip sync uv.lock might be more appropriate if uv.lock is comprehensive.
# Let's try a common pattern, assuming pyproject.toml lists dependencies.
RUN uv pip install --system .

# 6. Copy the rest of the application's code into the container
COPY . .

# 7. Install JupyterLab
RUN uv pip install --system jupyterlab

# 8. Expose port 8888 to the Docker host
EXPOSE 8888

# 9. Define the command to run JupyterLab
# --ip=0.0.0.0 makes the server accessible from outside the container
# --port=8888 specifies the port
# --no-browser prevents JupyterLab from trying to open a browser in the container
# --allow-root is often necessary when running in Docker containers
# --NotebookApp.token='' disables token authentication for convenience (be mindful of security implications)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"] 