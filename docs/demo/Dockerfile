# VHS (terminal recorder) plus the two tools score.sh needs. Built and run by `make demo-gif`.
FROM ghcr.io/charmbracelet/vhs:v0.10.0
# The image is Debian bookworm but tracks the "stable" suite, which has moved on: pin it back.
RUN sed -i 's/\bstable\b/bookworm/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl jq \
    && rm -rf /var/lib/apt/lists/*
