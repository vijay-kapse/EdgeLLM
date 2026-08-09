variable "region" {
  description = "AWS region to launch in. GPU capacity and AMI availability both vary by region."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = <<-EOT
    GPU instance type. g5.xlarge (1x A10G, 24 GB) is the cheapest type that fits
    GPTQ/AWQ INT4 quantization of a small model comfortably. g4dn.xlarge (1x T4,
    16 GB) is cheaper still but Turing-era, so INT4 kernel support is patchier.
  EOT
  type        = string
  default     = "g5.xlarge"
}

variable "key_name" {
  description = "Name of an existing EC2 key pair for SSH. No default: you must supply your own."
  type        = string
}

variable "allowed_ssh_cidr" {
  description = <<-EOT
    CIDR permitted to reach port 22. No default on purpose -- set it to your own
    address (e.g. "203.0.113.4/32"), not 0.0.0.0/0. Find yours with:
      curl -s https://checkip.amazonaws.com
  EOT
  type        = string

  validation {
    condition     = var.allowed_ssh_cidr != "0.0.0.0/0"
    error_message = "Refusing to open SSH to the entire internet. Pass your own /32."
  }
}

variable "root_volume_gb" {
  description = "Root EBS size. CUDA, torch, and model artifacts consume well over the 8 GB default."
  type        = number
  default     = 200
}

variable "use_spot" {
  description = <<-EOT
    Request a spot instance instead of on-demand. Substantially cheaper, at the
    cost of possible interruption -- fine for a benchmark run you can restart,
    not for anything you would be upset to lose mid-flight.
  EOT
  type        = bool
  default     = false
}

variable "repo_url" {
  description = "Git URL cloned onto the instance at boot."
  type        = string
  default     = "https://github.com/vijay-kapse/EdgeLLM.git"
}

variable "ami_name_filter" {
  description = <<-EOT
    AMI name pattern. Defaults to the AWS Deep Learning Base OSS Nvidia Driver
    AMI, which ships the NVIDIA driver, Docker, and the NVIDIA container toolkit
    preinstalled -- avoiding a fragile driver build in user_data. Verify the
    exact name is published in your region before relying on it.
  EOT
  type        = string
  default     = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"
}

variable "tags" {
  description = "Tags applied to every created resource."
  type        = map(string)
  default = {
    Project   = "EdgeLLM"
    ManagedBy = "terraform"
  }
}
