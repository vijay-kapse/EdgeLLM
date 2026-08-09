output "instance_id" {
  description = "EC2 instance ID."
  value       = aws_instance.bench.id
}

output "public_ip" {
  description = "Public IPv4 address of the benchmark host."
  value       = aws_instance.bench.public_ip
}

output "ami_id" {
  description = "Resolved AMI. Record this next to any published measurement."
  value       = data.aws_ami.gpu.id
}

output "ami_name" {
  description = "Resolved AMI name, so the driver/runtime provenance is recoverable later."
  value       = data.aws_ami.gpu.name
}

output "ssh_command" {
  description = "Ready-to-paste SSH command."
  value       = "ssh ubuntu@${aws_instance.bench.public_ip}"
}

output "bootstrap_log_command" {
  description = "Follow first-boot provisioning; the image build takes several minutes."
  value       = "ssh ubuntu@${aws_instance.bench.public_ip} 'tail -f /var/log/edgellm-bootstrap.log'"
}

output "fetch_results_command" {
  description = "Copy measurements off the box before destroying it."
  value       = "scp -r ubuntu@${aws_instance.bench.public_ip}:~/edgellm-host-info ./host-info-gpu"
}

output "reminder" {
  description = "Standing reminder."
  value       = "Run 'terraform destroy' when finished -- ${var.instance_type} bills hourly for as long as it exists."
}
