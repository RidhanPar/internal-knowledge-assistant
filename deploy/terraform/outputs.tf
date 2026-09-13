output "public_ip" {
  description = "Public IP of the instance."
  value       = aws_instance.app.public_ip
}

output "app_url" {
  description = "Base URL of the running API."
  value       = "http://${aws_instance.app.public_ip}"
}

output "health_url" {
  description = "Health endpoint to poll after boot."
  value       = "http://${aws_instance.app.public_ip}/health"
}

output "ssh_command" {
  description = "SSH in for debugging (uses the private key that matches ssh_public_key)."
  value       = "ssh ec2-user@${aws_instance.app.public_ip}"
}
