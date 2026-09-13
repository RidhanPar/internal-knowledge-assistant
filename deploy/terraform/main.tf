# One EC2 instance in the default VPC, with a security group and an SSH key.
# The instance's startup script installs Docker, clones the repo, writes .env,
# and runs the app plus Postgres with docker compose.

# --- Look up the network and image instead of hardcoding ids ---
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Latest Amazon Linux 2023 x86_64 image, resolved at plan time.
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }
  filter {
    name   = "state"
    values = ["available"]
  }
}

# --- Firewall: HTTP open to the world, SSH limited to a CIDR ---
resource "aws_security_group" "app" {
  name        = "${var.project_name}-sg"
  description = "Allow HTTP from anywhere and SSH from an allowed CIDR"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Project = var.project_name }
}

resource "aws_key_pair" "app" {
  key_name   = "${var.project_name}-key"
  public_key = var.ssh_public_key
}

resource "aws_instance" "app" {
  ami                         = data.aws_ami.al2023.id
  instance_type               = var.instance_type
  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.app.id]
  key_name                    = aws_key_pair.app.key_name
  associate_public_ip_address = true

  # The API key ends up in user_data, which is readable by anyone who can
  # describe the instance. Acceptable for a demo; a production setup would store
  # it in SSM Parameter Store (SecureString) and give the instance a role to
  # read it. That path needs IAM permissions this deploy intentionally avoids.
  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    repo_url               = var.repo_url
    anthropic_api_key      = var.anthropic_api_key
    anthropic_workspace_id = var.anthropic_workspace_id
    anthropic_model        = var.anthropic_model
  })
  # Re-run user_data if it changes (forces replacement on script edits).
  user_data_replace_on_change = true

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  tags = { Name = var.project_name, Project = var.project_name }
}
