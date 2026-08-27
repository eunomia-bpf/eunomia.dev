FROM mcr.microsoft.com/devcontainers/javascript-node:1-22-bookworm@sha256:0d29e5fdc64f8397cd502223e0c4679f1e60877ca0fd2db4f2e2e0028e4271af

ARG GH_CLI_VERSION=2.98.0

USER root

RUN set -eux; \
    architecture="$(dpkg --print-architecture)"; \
    case "$architecture" in \
      amd64) gh_sha256="3b8ac6b30336802fc1a858d7c084e11cdf24ac1a761ca90b68022d7d729208de" ;; \
      arm64) gh_sha256="cf689084f3a3618f7eae4a2420d335d74626d65f5e594b9828d125d69f800d86" ;; \
      *) echo "Unsupported architecture: $architecture" >&2; exit 1 ;; \
    esac; \
    archive="gh_${GH_CLI_VERSION}_linux_${architecture}.tar.gz"; \
    curl --fail --location --show-error \
      --output "/tmp/${archive}" \
      "https://github.com/cli/cli/releases/download/v${GH_CLI_VERSION}/${archive}"; \
    echo "${gh_sha256}  /tmp/${archive}" | sha256sum --check --strict; \
    tar --extract --gzip --file "/tmp/${archive}" --directory /tmp; \
    install --mode 0755 \
      "/tmp/gh_${GH_CLI_VERSION}_linux_${architecture}/bin/gh" \
      /usr/local/bin/gh; \
    rm --recursive --force "/tmp/${archive}" \
      "/tmp/gh_${GH_CLI_VERSION}_linux_${architecture}"; \
    npm install --global @openai/codex@0.147.0; \
    gh --version; \
    codex --version
