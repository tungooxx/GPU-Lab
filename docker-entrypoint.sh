#!/bin/sh
set -eu

# Windows bind mounts do not preserve Unix key permissions. Copy the mounted
# key into the container with restrictive permissions so OpenSSH can use it.
if [ -s /run/ssh-key ]; then
    install -d -m 700 /root/.ssh
    install -m 600 /run/ssh-key /root/.ssh/gpu_lab_ed25519
    cat > /root/.ssh/config <<'EOF'
Host github.com
    IdentityFile /root/.ssh/gpu_lab_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
EOF
    chmod 600 /root/.ssh/config
fi

exec "$@"
