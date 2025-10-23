# -----------------------------------------------------------------------------
# Base image
# -----------------------------------------------------------------------------
FROM dongwxxkchoi/habitat:2.0
# workspace and data directory
RUN mkdir -p /workspace /data

# -----------------------------------------------------------------------------
# Install system prerequisites
# -----------------------------------------------------------------------------
USER root

# Clean up unnecessary files and HabitatLLM contents
RUN rm -rf /home/* /root/.bash_history /root/.cache /root/.ssh
RUN rm -rf /HabitatLLM/*


# -----------------------------------------------------------------------------
# Copy repository codes and mount volumes
# -----------------------------------------------------------------------------
WORKDIR /HabitatLLM

COPY . /HabitatLLM

WORKDIR /HabitatLLM

SHELL ["conda", "run", "-n", "habitat", "/bin/bash", "-c"]