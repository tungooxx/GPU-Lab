FROM node:22.18.0-bookworm-slim

ARG PAPER2AGENT_COMMIT=e573687e15f345e3f375cd0851373d588e436be3
ARG CLAUDE_CODE_VERSION=2.1.227

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"

RUN git clone --filter=blob:none https://github.com/jmiao24/Paper2Agent.git /opt/paper2agent \
    && git -C /opt/paper2agent checkout --detach "${PAPER2AGENT_COMMIT}" \
    && test "$(git -C /opt/paper2agent rev-parse HEAD)" = "${PAPER2AGENT_COMMIT}"

RUN python3 -m venv /opt/worker \
    && /opt/worker/bin/pip install --no-cache-dir \
       "mcp[cli]>=1.27,<2" \
       "pydantic>=2.10,<3" "starlette>=0.41,<1" "uvicorn>=0.34,<1"

COPY src/gpu_lab /opt/gpu-lab/gpu_lab
RUN useradd --create-home --uid 10002 paperagent \
    && mkdir -p /opt/paper2agent/projects /home/paperagent/.claude \
    && chown -R paperagent:paperagent /opt/paper2agent/projects /home/paperagent

ENV PATH="/opt/worker/bin:${PATH}" \
    PYTHONPATH=/opt/gpu-lab \
    PAPER2AGENT_ROOT=/opt/paper2agent \
    PAPER2AGENT_UPSTREAM_COMMIT=${PAPER2AGENT_COMMIT} \
    CLAUDE_CONFIG_DIR=/home/paperagent/.claude
USER paperagent
CMD ["python", "-m", "gpu_lab.paper2agent_worker"]
