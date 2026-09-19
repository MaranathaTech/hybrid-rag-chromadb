#!/usr/bin/env bash
#
# Build the image and deploy the whole stack to a local Kubernetes cluster.
#
#   ./run.sh           build, deploy, wait for ready, smoke test
#   ./run.sh --test    run the smoke test against an already-deployed stack
#   ./run.sh --logs    tail the API logs
#   ./run.sh --down    delete the namespace
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="hybrid-rag"
IMAGE_NAME="hybrid-rag:local"
K8S_DIR="$SCRIPT_DIR/k8s"
PORT_FORWARD_PID=""

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

log()   { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[x]${NC} $*" >&2; }
info()  { echo -e "${CYAN}[i]${NC} $*"; }

cleanup() {
    if [[ -n "$PORT_FORWARD_PID" ]] && kill -0 "$PORT_FORWARD_PID" 2>/dev/null; then
        kill "$PORT_FORWARD_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- Smoke test ---------------------------------------------------------------
# Deploying successfully is not the same as working. This queries the running
# service and checks the answer is actually correct, not merely well-formed.
smoke_test() {
    log "Port-forwarding to the API..."
    kubectl port-forward -n "$NAMESPACE" svc/api 18080:8080 >/dev/null 2>&1 &
    PORT_FORWARD_PID=$!

    local attempt=0
    until curl -sf "http://localhost:18080/ready" >/dev/null 2>&1; do
        attempt=$((attempt + 1))
        if (( attempt > 30 )); then
            error "API did not become ready through the port-forward."
            return 1
        fi
        sleep 2
    done

    local ready_body
    ready_body="$(curl -s http://localhost:18080/ready)"
    info "ready: $ready_body"

    log "Query 1/2 — exact identifier (expects vector-dimension-mismatch)"
    local exact_top
    exact_top="$(curl -s -X POST http://localhost:18080/search \
        -H 'Content-Type: application/json' \
        -d '{"query":"E1043","limit":3,"strategy":"rrf"}' \
        | python3 -c 'import sys,json; d=json.load(sys.stdin)["documents"]; print(d[0]["id"] if d else "NONE")')"

    if [[ "$exact_top" != "vector-dimension-mismatch" ]]; then
        error "Expected 'vector-dimension-mismatch', got '$exact_top'"
        return 1
    fi
    log "  -> $exact_top"

    log "Query 2/2 — paraphrase, no shared terms (expects k8s-replicas)"
    local para_top
    para_top="$(curl -s -X POST http://localhost:18080/search \
        -H 'Content-Type: application/json' \
        -d '{"query":"how do I keep my service up when a machine dies","limit":3,"strategy":"rrf"}' \
        | python3 -c 'import sys,json; d=json.load(sys.stdin)["documents"]; print(d[0]["id"] if d else "NONE")')"

    if [[ "$para_top" != "k8s-replicas" ]]; then
        error "Expected 'k8s-replicas', got '$para_top'"
        return 1
    fi
    log "  -> $para_top"

    log "Smoke test passed: retrieval is correct on both query types."
    return 0
}

# --- Subcommands --------------------------------------------------------------
case "${1:-}" in
    --down)
        log "Deleting namespace $NAMESPACE..."
        kubectl delete namespace "$NAMESPACE" --ignore-not-found
        log "Done."
        exit 0
        ;;
    --logs)
        exec kubectl logs -n "$NAMESPACE" -l app.kubernetes.io/name=api -f --tail=100
        ;;
    --test)
        smoke_test
        exit $?
        ;;
esac

# --- Pre-flight ---------------------------------------------------------------
log "Checking prerequisites..."
for tool in docker kubectl python3 curl; do
    if ! command -v "$tool" &>/dev/null; then
        error "$tool not found on PATH."
        exit 1
    fi
done

if ! kubectl cluster-info &>/dev/null; then
    error "Cannot reach a Kubernetes cluster."
    error "Current context: $(kubectl config current-context 2>/dev/null || echo '<none>')"
    error ""
    error "A context can exist while the cluster behind it is stopped, which is"
    error "what a 'connection refused' on the API server port means."
    error "  Rancher Desktop: enable Kubernetes in Preferences, then wait for the"
    error "                   node to report Ready ('kubectl get nodes')."
    error "  Docker Desktop:  enable Kubernetes in Settings."
    error "  kind:            kind create cluster"
    error "  minikube:        minikube start"
    exit 1
fi
log "Cluster reachable: $(kubectl config current-context)"

# --- Build --------------------------------------------------------------------
# The unit tests run inside this build (see Dockerfile), so a failing test
# fails the image and nothing broken can reach the cluster.
log "Building $IMAGE_NAME (unit tests run inside the build)..."
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
log "Image built."

# A cluster that pulls from its own containerd (k3s/Rancher Desktop) sees the
# local image directly. kind and minikube need it loaded explicitly.
if command -v kind &>/dev/null && kubectl config current-context | grep -q '^kind-'; then
    log "kind detected — loading image into the cluster..."
    kind load docker-image "$IMAGE_NAME" --name "$(kubectl config current-context | sed 's/^kind-//')"
elif command -v minikube &>/dev/null && kubectl config current-context | grep -q 'minikube'; then
    log "minikube detected — loading image into the cluster..."
    minikube image load "$IMAGE_NAME"
fi

# --- Deploy -------------------------------------------------------------------
log "Applying manifests..."
kubectl apply -f "$K8S_DIR/namespace.yml"
kubectl apply -f "$K8S_DIR/configmap.yml"
kubectl apply -f "$K8S_DIR/chroma-service.yml"
kubectl apply -f "$K8S_DIR/chroma-statefulset.yml"

log "Waiting for Chroma to be ready..."
kubectl rollout status statefulset/chroma -n "$NAMESPACE" --timeout=180s

kubectl apply -f "$K8S_DIR/api-service.yml"
kubectl apply -f "$K8S_DIR/api-deployment.yml"

# Force a fresh pod when the image was rebuilt under an unchanged tag.
kubectl rollout restart deployment/api -n "$NAMESPACE" >/dev/null 2>&1 || true

log "Waiting for the API to be ready (first boot loads the embedding model)..."
if ! kubectl rollout status deployment/api -n "$NAMESPACE" --timeout=300s; then
    error "API failed to become ready. Recent logs:"
    kubectl logs -n "$NAMESPACE" -l app.kubernetes.io/name=api --tail=50 || true
    exit 1
fi

echo
kubectl get pods -n "$NAMESPACE"
echo

# --- Verify -------------------------------------------------------------------
if ! smoke_test; then
    error "Deployment is up but the smoke test failed."
    kubectl logs -n "$NAMESPACE" -l app.kubernetes.io/name=api --tail=50 || true
    exit 1
fi

echo
info "Try it yourself:"
info "  kubectl port-forward -n $NAMESPACE svc/api 8080:8080"
info "  curl 'http://localhost:8080/compare?query=E1043'"
info ""
info "  ./run.sh --logs     tail API logs"
info "  ./run.sh --down     tear everything down"
