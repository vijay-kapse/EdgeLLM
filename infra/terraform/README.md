# EdgeLLM GPU benchmark host (Terraform)

Stands up one GPU instance on AWS so the benchmark rows an Apple Silicon laptop
cannot produce can actually be measured.

## Why

Two gaps in the repo's benchmark table need CUDA hardware:

1. **GPTQ / AWQ INT4.** The current INT4 numbers come from ONNX Runtime's
   block-wise weight-only path. GPTQ and AWQ need CUDA, and `edgellm quantize`
   honestly reports them as unavailable on a Mac rather than faking a row.
2. **ONNX Runtime CUDA execution provider.** No NVIDIA device, no CUDA EP
   numbers.

Rather than leave both blank forever, this provisions the smallest sensible box
to fill them in.

## Prerequisites

- Terraform >= 1.5
- AWS credentials with EC2 permissions (`aws configure`, or `AWS_PROFILE`)
- An existing EC2 key pair in the target region
- GPU instance quota in that region. Fresh accounts frequently have a **zero**
  vCPU limit for G-family instances; if `apply` fails with `VcpuLimitExceeded`,
  request a quota increase for "Running On-Demand G and VT instances" and expect
  it to take a day or so.

## Use

```bash
cd infra/terraform
terraform init

MYIP="$(curl -s https://checkip.amazonaws.com)/32"

terraform plan \
  -var="key_name=YOUR_KEYPAIR" \
  -var="allowed_ssh_cidr=$MYIP"

terraform apply \
  -var="key_name=YOUR_KEYPAIR" \
  -var="allowed_ssh_cidr=$MYIP"
```

Then follow first boot — the container build takes several minutes:

```bash
terraform output -raw bootstrap_log_command | bash
```

When finished, copy results off **before** destroying:

```bash
terraform output -raw fetch_results_command | bash
terraform destroy -var="key_name=YOUR_KEYPAIR" -var="allowed_ssh_cidr=$MYIP"
```

## Cost

`g5.xlarge` on-demand runs on the order of **$1/hour** in `us-east-1`; a month of
forgetting to destroy it is roughly $700. Check current pricing before you
apply — these rates change.

`-var="use_spot=true"` cuts that substantially in exchange for possible
interruption, which is an easy trade for a restartable benchmark.

## What the bootstrap does and does not do

**Does:** records host CPU/GPU/memory provenance, verifies Docker and the NVIDIA
driver, clones the repo, builds the container image, and runs the INT8 GEMM
microbenchmark — which compiles to the **AVX2** path here, a genuinely different
kernel from the NEON SDOT path measured on Apple Silicon.

**Does not:** run the model quantization or benchmark suite unattended. Those
steps are long, download-heavy, and fail in interesting ways. A number produced
by an unattended script nobody watched is exactly what this repo refuses to
publish. SSH in and run them yourself; `~/edgellm-host-info/NEXT_STEPS.txt` on
the instance has the commands.

## Security notes

- `allowed_ssh_cidr` has **no default** and rejects `0.0.0.0/0`. Supply your own
  address.
- IMDSv2 is required (`http_tokens = "required"`).
- The root volume is encrypted and deleted on termination.
- The security group permits unrestricted egress, which package registries,
  GitHub, and Hugging Face all need.

## Verification status

The HCL here has **not** been run through `terraform validate` or `apply` —
Terraform is not installed on the machine where these files were written. The
`user_data.sh` template was rendered and checked with `bash -n`. Run
`terraform init && terraform validate` before trusting it, and read the `plan`
output before the first `apply`.
