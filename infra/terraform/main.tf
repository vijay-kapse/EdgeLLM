# EdgeLLM GPU benchmark box.
#
# Why this exists: the benchmark table in the repo README has holes that an
# Apple Silicon laptop cannot fill. GPTQ/AWQ-style INT4 quantization needs CUDA,
# and the ONNX Runtime CUDA execution provider needs an NVIDIA device. This
# stands up the smallest sensible GPU instance so those rows can be measured
# rather than left blank -- which is the repo's standing rule.
#
# Usage:
#   terraform init
#   terraform plan  -var="key_name=my-key" -var="allowed_ssh_cidr=$(curl -s https://checkip.amazonaws.com)/32"
#   terraform apply -var="key_name=my-key" -var="allowed_ssh_cidr=..."
#   terraform destroy -var="key_name=my-key" -var="allowed_ssh_cidr=..."
#
# This instance bills by the hour while it exists. Destroy it when the run is
# done; a forgotten g5.xlarge is roughly $700/month.

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_ami" "gpu" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = [var.ami_name_filter]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

data "aws_vpc" "default" {
  default = true
}

resource "aws_security_group" "bench" {
  name_prefix = "edgellm-bench-"
  description = "SSH in from one operator address; unrestricted egress for package and model downloads."
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH from the operator only"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    description = "Package registries, GitHub, Hugging Face"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, { Name = "edgellm-bench" })

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_instance" "bench" {
  ami           = data.aws_ami.gpu.id
  instance_type = var.instance_type
  key_name      = var.key_name

  vpc_security_group_ids = [aws_security_group.bench.id]

  root_block_device {
    volume_size           = var.root_volume_gb
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  # Spot is opt-in: cheaper, interruptible. Omitted entirely when use_spot is
  # false so the instance is plain on-demand rather than a one-time spot request.
  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []

    content {
      market_type = "spot"

      spot_options {
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  # IMDSv2 required; the metadata endpoint is a standard SSRF pivot.
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    repo_url = var.repo_url
  })

  # Changing user_data should rebuild the box -- a half-provisioned benchmark
  # host is worse than a fresh one.
  user_data_replace_on_change = true

  tags = merge(var.tags, { Name = "edgellm-bench" })
}
