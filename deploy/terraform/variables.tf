# Inputs to the deploy. Values with no default are supplied in terraform.tfvars,
# which is gitignored because it holds the API key.

variable "region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "eu-north-1"
}

variable "instance_type" {
  description = "EC2 instance size. t3.medium (4 GB) fits the local embedding model plus Postgres."
  type        = string
  default     = "t3.medium"
}

variable "project_name" {
  description = "Name tag and key-pair name prefix."
  type        = string
  default     = "internal-knowledge-assistant"
}

variable "repo_url" {
  description = "Public git repo the instance clones at boot."
  type        = string
  default     = "https://github.com/RidhanPar/internal-knowledge-assistant.git"
}

variable "ssh_public_key" {
  description = "SSH public key registered on the instance for admin access."
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to reach SSH (port 22). Narrow this to your IP for real use."
  type        = string
  default     = "0.0.0.0/0"
}

# --- Application secrets and settings (written to .env on the instance) ---
variable "anthropic_api_key" {
  description = "Anthropic API key. Sensitive; never commit."
  type        = string
  sensitive   = true
}

variable "anthropic_workspace_id" {
  description = "Anthropic workspace id (needed for organization-scoped keys)."
  type        = string
  default     = ""
}

variable "anthropic_model" {
  description = "Anthropic model id used for generation and the eval judge."
  type        = string
  default     = "claude-sonnet-5"
}
