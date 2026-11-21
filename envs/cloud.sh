# Google Cloud SDK as installed by scripts/setup_remote.sh (into $LOOP_STEER_TOOLS, default /workspace/tools), with
# its config kept on the persistent volume. Source in every shell that talks to GCS (non-interactive shells do not
# read ~/.bashrc). A machine with the cloud-sync tooling (/workspace/cloud-sync/env.sh) uses that instead.
# Set GCS_BUCKET (gs://...) yourself; the account is whatever `gcloud auth login` made active.
if [ -f /workspace/cloud-sync/env.sh ]; then
  source /workspace/cloud-sync/env.sh
else
  _tools=${LOOP_STEER_TOOLS:-/workspace/tools}
  export CLOUDSDK_CONFIG=${CLOUDSDK_CONFIG:-/workspace/.machine/gcloud}
  export CLOUDSDK_CORE_DISABLE_USAGE_REPORTING=true CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED=false
  export PATH="$_tools/google-cloud-sdk/bin:$PATH"
  unset _tools
fi
