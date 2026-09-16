# Agentic Code Generation — DeepSeek-V3 on AMD Instinct MI355X

## Overview

This guide deploys [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) (671B MoE,
37B active) on AMD Instinct MI355X nodes, P/D-disaggregated with wide expert parallelism via
LeaderWorkerSets. It composes the [wide-ep-lws](../wide-ep-lws/README.md) DeepSeek-V3 recipe
(`modelserver/amd/vllm-deepseek-v3/`) with the agentic routing and KV-offloading configuration.
Each layer relieves a specific pressure of the agentic code-generation workload (deep multi-turn
sessions over repository-scale contexts; see the [Agentic Serving overview](README.md)):

- **Wide EP (DEP16)** — the MoE experts are spread across ranks (TP=1, DP=EP=16) rather than
  tensor-parallel sharded, and DeepSeek-V3's MLA keeps the per-token KV footprint small; together
  they preserve HBM for KV cache, and on this workload cache hit rate — not FLOPs — sets throughput.
- **P/D disaggregation (MoRI-IO KV transfer)** — input processing and token generation use
  separate pools. Prefill runs MoRI-EP high-throughput all-to-all; decode runs low-latency. The
  prefill-to-decode KV transfer uses MoRI-IO in WRITE mode over a rail-only RoCE fabric, in place
  of DeepEP and NIXL.
- **Dual-tier prefix-cache routing** — the EPP scores both GPU- and CPU-resident prefix caches
  when picking an endpoint, so resumed sessions land where their context can be restored rather
  than recomputed.
- **CPU KV offloading** — CPU memory keeps evicted prefixes reusable after HBM fills; on prefill
  nodes with local NVMe, the tiered variant adds a lower NVMe tier.

The model server manifests live under the [wide-ep-lws guide](../wide-ep-lws/README.md); this
guide composes its `deployments/offload-cpu` overlay (fabric + CPU offload already bundled) with
the agentic dual-scorer router config. Prefix caching is on and the context window is the full
163 840 tokens.

> [!CAUTION]
> **Experimental.** The MoRI-IO + `OffloadingConnector` `MultiConnector` composition that backs
> CPU/tiered KV offloading has not yet been validated on the ROCm engine image. The recipe runs
> as plain wide-EP P/D with offloading off (`OFFLOADING_MODE=off`) as its byte-for-byte default;
> this guide opts into `OFFLOADING_MODE=cpu`. Treat the offloading path as unhardened and validate
> on your cluster before relying on it. The engine and sidecar images are personal builds
> (`ghcr.io/vcave/...`); see the [recipe README](../wide-ep-lws/modelserver/amd/vllm-deepseek-v3/README.md#images).

## Default Configuration

The default deployment is `deployments/offload-cpu` — 2 prefill + 2 decode DEP16 engines
(2P2D, 32 GPUs on 4 nodes) with CPU KV offloading on prefill.

| Parameter           | Value                                                                          |
| ------------------- | ------------------------------------------------------------------------------ |
| Model               | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) (688.7 GB, 163 shards) |
| Accelerator         | AMD Instinct MI355X (8 GPUs per node, 4 nodes)                                 |
| Serving topology    | P/D disaggregated — 1 prefill role + 1 decode role, each `replicas: 1` `size: 2`, TP=1, DP=16, EP=16 (DEP16, 2 pods / node-pair) |
| Expert all-to-all   | MoRI-EP (`mori_high_throughput` prefill / `mori_low_latency` decode)           |
| KV transfer         | MoRIIOConnector (WRITE mode), rail-only RoCE fabric                            |
| KV cache offloading | CPU on prefill (`offloading-cpu` component, `OFFLOADING_MODE=cpu`)             |
| Prefix caching      | On                                                                             |
| Max model length    | 163 840                                                                        |
| Routing             | Dual GPU+CPU prefix-cache scoring ([`agentic-serving-amd.values.yaml`](router/agentic-serving-amd.values.yaml)) |

For prefill nodes with local NVMe, point the model-server overlay at
`deployments/offload-tiered` (CPU + NVMe) instead — see the
[recipe README](../wide-ep-lws/modelserver/amd/vllm-deepseek-v3/README.md#deploy-the-model-server).

### Supported Hardware Backends

| Backend              | Directory                              | Notes                                                                 |
| -------------------- | -------------------------------------- | --------------------------------------------------------------------- |
| AMD Instinct (vLLM)  | `modelserver/amd/vllm/deepseek-v3/`    | Composes `wide-ep-lws/modelserver/amd/vllm-deepseek-v3/` (MI355X, 2P2D wide-EP, MoRI-EP + MoRI-IO, rail-only RoCE fabric) |

## Prerequisites

In addition to the general [agentic-serving](README.md) and
[wide-ep-lws](../wide-ep-lws/README.md#prerequisites) prerequisites:

- Installed client tools (`kubectl`, `helm`).
- **32 AMD Instinct MI355X GPUs on four nodes**, 8 per node, and an **8-rail RDMA fabric**
  (`amd.com/vnic`, one Multus attachment per rail, every rail reachable between every pair of the
  four nodes). `providers/amd-ci` is a worked fabric overlay; copy and re-point it (NAD name,
  namespace, `MORI_RDMA_DEVICES`, RoCE service level / traffic class) for another cluster. See the
  [recipe README](../wide-ep-lws/modelserver/amd/vllm-deepseek-v3/README.md#prerequisites).
- **A pre-provisioned `model-pvc` claim** in the target namespace, `ReadWriteMany`, `1Ti`
  recommended (the model is 688.7 GB). No overlay creates it. It may start empty; the first pod
  downloads the weights via `HF_HOME` on the claim, later pods and runs read from the volume.
- Set the environment variables:

  ```bash
  export REPO_ROOT=$(realpath $(git rev-parse --show-toplevel))
  source ${REPO_ROOT}/guides/env.sh
  export GUIDE_NAME="agentic-serving"
  export NAMESPACE=llm-d-agentic-serving
  ```

- Install the Gateway API Inference Extension CRDs:

  ```bash
  kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/${GAIE_URL}/v1-manifests.yaml
  ```

- Create the target namespace:

  ```bash
  kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
  ```

- [Create the `llm-d-hf-token` secret in your target namespace with the key `HF_TOKEN`](../../helpers/hf-token.md).
<!-- llm-d-cicd:skip start -->
  ```bash
  export HF_TOKEN=<your HuggingFace token>
  kubectl create secret generic llm-d-hf-token \
    --from-literal="HF_TOKEN=${HF_TOKEN}" \
    --namespace "${NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -
  ```
<!-- llm-d-cicd:skip end -->

- Deploy the [LeaderWorkerSet controller](https://lws.sigs.k8s.io/docs/installation/) (each role
  is a LeaderWorkerSet).

## Installation Instructions

### 1. Deploy the llm-d Router

The router layers the AMD agentic overrides
([`agentic-serving-amd.values.yaml`](router/agentic-serving-amd.values.yaml)) on the wide-ep-lws
values and the AMD accelerator override: a GPU prefix-cache scorer (weight 5), a CPU
prefix-cache scorer (weight 2, fixed-capacity LRU tracking offloaded blocks), and an
active-request scorer for load balancing. `amd.values.yaml` sets the AMD model-server match
labels and is required, not tuning:

```bash
helm install ${GUIDE_NAME} \
    ${ROUTER_STANDALONE_CHART} \
    -f ${REPO_ROOT}/guides/recipes/router/base.values.yaml \
    -f ${REPO_ROOT}/guides/wide-ep-lws/router/wide-ep-lws.values.yaml \
    -f ${REPO_ROOT}/guides/wide-ep-lws/router/amd.values.yaml \
    -f ${REPO_ROOT}/guides/${GUIDE_NAME}/router/agentic-serving-amd.values.yaml \
    -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}
```

### 2. Deploy the Model Server

Apply the Kustomize overlay for the default deployment (`offload-cpu`):

```bash
export MODEL=deepseek-ai/DeepSeek-V3
kubectl apply -n ${NAMESPACE} -k ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/amd/vllm/deepseek-v3/
```

The overlay bundles the `providers/amd-ci` fabric overlay; for another cluster, copy and edit
`guides/wide-ep-lws/modelserver/amd/vllm-deepseek-v3/providers/amd-ci/` first (at minimum the NAD
name and namespace, the `MORI_RDMA_DEVICES` names, and the RoCE service level and traffic class).

Startup takes tens of minutes: 688.7 GB of weights across 8 local ranks per pod, a cross-pod DP
rendezvous, AITER JIT compilation and graph capture. The first run also downloads the model. Wait
for pods to become ready:

```bash
kubectl get pods -n ${NAMESPACE} -l llm-d.ai/model=DeepSeek-V3 -w
```

## Verification

### 1. Get the IP of the Proxy

```bash
export IP=$(kubectl get service ${GUIDE_NAME}-epp -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
```

### 2. Send Test Requests

Open a temporary interactive shell inside the cluster:

```bash
kubectl run curl-debug --rm -it \
    --image=cfmanteiga/alpine-bash-curl-jq \
    --env="IP=$IP" \
    --env="NAMESPACE=$NAMESPACE" \
    -- /bin/bash
```

Send a completion request:

```bash
curl -X POST http://${IP}/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "deepseek-ai/DeepSeek-V3",
        "prompt": "Explain how a simple agent loop works in 3 sentences."
    }' | jq
```

Under load, all 16 engines should be non-zero on both roles; `4 of 8` is the signature of a
routing sidecar missing PR #2816 (see the
[recipe verification notes](../wide-ep-lws/modelserver/amd/vllm-deepseek-v3/README.md#verification)).

## Benchmarking

An [`inference-perf`](https://github.com/kubernetes-sigs/inference-perf) preset for this
deployment is in [`benchmark-templates/agentic-serving-deepseek-v3.yaml`](benchmark-templates/agentic-serving-deepseek-v3.yaml),
driven through [`llm-d-benchmark`](https://github.com/llm-d/llm-d-benchmark) with the agentic
conversation-replay workload (large reused contexts, bursty locality-heavy traffic). Published
benchmark numbers for MI355X are pending cluster time.

## Cleanup

```bash
helm uninstall ${GUIDE_NAME} -n ${NAMESPACE}
kubectl delete -n ${NAMESPACE} -k ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/amd/vllm/deepseek-v3/
```

`model-pvc` is not part of any overlay, so it survives this and the next run reuses the
downloaded weights.

<!-- llm-d-cicd:skip start -->
```bash
kubectl delete namespace ${NAMESPACE}
```
<!-- llm-d-cicd:skip end -->
