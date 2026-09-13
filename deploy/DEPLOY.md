# Deploy runbook (AWS EC2 with Terraform)

This is the operational companion to the deploy section in the main README. The
Terraform is in `deploy/terraform/`.

## What it creates

- One EC2 instance in the default VPC, with a public IP.
- A security group: port 80 open to the world, port 22 limited to a CIDR you set.
- An SSH key pair.

The instance startup script installs Docker, clones this repo, writes `.env`,
runs `docker-compose.prod.yml` (the API plus pgvector), and ingests the corpus
once.

## Cost

The instance bills for as long as it exists. Left running, expect a small number
of tens of US dollars per month for the size used here. Anthropic usage is billed
per token on top and is cents for a demo. Destroy the stack when you are done.

## First deploy

```bash
# 1. Create an SSH key and your tfvars.
ssh-keygen -t ed25519 -f deploy_key -N ""
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
# Fill terraform.tfvars with your ANTHROPIC_API_KEY, workspace id if needed, and
# the contents of deploy_key.pub. Optionally set allowed_ssh_cidr to your IP.

# 2. Provision.
terraform init
terraform apply

# 3. Wait for the stack to build and ingest, then check health.
#    corpus_chunks above zero means ingestion finished.
curl http://<public_ip>/health
```

## Redeploying after code changes

The instance clones the GitHub repo at boot, so push your changes first, then
recreate the instance so its startup script runs again:

```bash
git push origin main
cd deploy/terraform
terraform apply -replace=aws_instance.app
```

## Debugging

```bash
ssh -i deploy_key ec2-user@<public_ip>
sudo cat /var/log/cloud-init-output.log        # startup script output
cd /opt/app && sudo docker compose -f docker-compose.prod.yml ps
sudo docker compose -f docker-compose.prod.yml logs app
```

## Teardown (stops all cost)

```bash
cd deploy/terraform
terraform destroy
```

Confirm afterwards that the instance no longer appears in the EC2 console for the
region.

## Notes on access this account needed

- The IAM user needs EC2 permissions (launch instances, security groups, key
  pairs). AmazonEC2FullAccess works, or a scoped inline policy covering
  RunInstances, TerminateInstances, the security group actions, and ImportKeyPair.
- This account is restricted to free-tier-eligible instance types, so the deploy
  uses `c7i-flex.large` (4 GB), which is on that allowed set and large enough for
  the local embedding model plus Postgres.
