#cloud-config
# Runs automatically on first boot via Azure custom_data.
# Installs deps, downloads code, and starts the experiment as a systemd service.

package_update: true
package_upgrade: false
packages:
  - python3-venv
  - python3-pip
  - unzip
  - curl
  - tmux
  - htop

write_files:
  # ── .env for the experiment ──────────────────────────────────────────────────
  - path: /opt/stategen/.env
    permissions: "0600"
    content: |
      LLM_PROVIDER=openai_compatible
      LLM_MODEL=${model}
      LLM_BASE_URL=${base_url}
      CMU_API_KEY=${cmu_api_key}
      DEEPSEEK_API_KEY=${cmu_api_key}
      MAX_NEW_TOKENS=${max_new_tokens}
      MPLBACKEND=Agg

  # ── systemd service: runs the experiment, uploads results when done ──────────
  - path: /etc/systemd/system/stategen-experiment.service
    permissions: "0644"
    content: |
      [Unit]
      Description=StateGen Experiment (${experiment_name} / ${model})
      After=network-online.target cloud-final.service
      Wants=network-online.target

      [Service]
      Type=simple
      WorkingDirectory=/opt/stategen
      EnvironmentFile=/opt/stategen/.env
      ExecStart=/opt/stategen/.venv/bin/python experiments/run_bigcodebench.py \
          --method ${methods_str} \
          --provider openai_compatible \
          --model ${model} \
          --base_url ${base_url} \
          --api_key ${cmu_api_key} \
          --experiment ${experiment_name} \
          ${bcb_flag}
      StandardOutput=append:/opt/stategen/experiment.log
      StandardError=append:/opt/stategen/experiment.log
      # On success, upload results to Azure Blob
      ExecStartPost=/opt/stategen/upload_results.sh
      Restart=no
      TimeoutStartSec=0

      [Install]
      WantedBy=multi-user.target

  # ── Result uploader: runs after the experiment service exits successfully ─────
  - path: /opt/stategen/upload_results.sh
    permissions: "0755"
    content: |
      #!/bin/bash
      # Upload results to Azure Blob Storage after experiment completes.
      set -e
      STORAGE="${storage_account}"
      SAS="${results_sas}"
      RESULTS_DIR="/opt/stategen/results/${experiment_name}"
      DEST="https://$${STORAGE}.blob.core.windows.net/results/${experiment_name}"

      echo "[upload] Uploading results from $${RESULTS_DIR} ..."
      for f in "$${RESULTS_DIR}"/*.jsonl "$${RESULTS_DIR}"/*.json; do
          [ -f "$f" ] || continue
          fname=$(basename "$f")
          curl -s -X PUT \
            -H "x-ms-blob-type: BlockBlob" \
            -H "x-ms-date: $(date -u +%a,\ %d\ %b\ %Y\ %H:%M:%S\ GMT)" \
            --data-binary @"$f" \
            "$${DEST}/$${fname}?$${SAS}" && echo "[upload] OK: $${fname}"
      done
      echo "[upload] Done."

  # ── Status checker you can run manually ─────────────────────────────────────
  - path: /opt/stategen/status.sh
    permissions: "0755"
    content: |
      #!/bin/bash
      echo "=== Experiment status ==="
      systemctl is-active stategen-experiment && echo "RUNNING" || echo "STOPPED/DONE"
      echo ""
      echo "=== Last 30 log lines ==="
      tail -30 /opt/stategen/experiment.log 2>/dev/null || echo "(log not yet created)"
      echo ""
      echo "=== Result files ==="
      ls -lh /opt/stategen/results/${experiment_name}/ 2>/dev/null || echo "(no results yet)"

runcmd:
  # Download and extract code (-o = overwrite without prompting)
  - curl -s -o /tmp/stategen.zip '${code_url}'
  - unzip -qo /tmp/stategen.zip -d /opt/stategen
  - rm /tmp/stategen.zip

  # Set up Python virtual environment
  - python3 -m venv /opt/stategen/.venv
  - /opt/stategen/.venv/bin/pip install --quiet --upgrade pip
  # CPU-only torch (~200MB vs ~800MB GPU build) — no GPU on these VMs
  - /opt/stategen/.venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  - /opt/stategen/.venv/bin/pip install --quiet -r /opt/stategen/requirements.txt
  # bigcodebench installed from PyPI via requirements.txt — no submodule needed
  # execution_engine calls 'python' but Ubuntu only has 'python3'
  - ln -sf /usr/bin/python3 /usr/local/bin/python

  # Create results directory
  - mkdir -p /opt/stategen/results/${experiment_name}

  # Enable and start the experiment service
  - systemctl daemon-reload
  - systemctl enable stategen-experiment.service
  - systemctl start stategen-experiment.service

  - echo "cloud-init complete — experiment service started"
