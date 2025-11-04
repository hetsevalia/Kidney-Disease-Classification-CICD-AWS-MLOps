# GitHub Actions Workflow - CI/CD Pipeline

## 📋 Overview

This workflow implements a complete CI/CD pipeline for the Kidney Disease Classification project:

1. **Continuous Integration** - Code validation and testing
2. **Build & Push** - Docker image build and push to AWS ECR
3. **Continuous Deployment** - Automated deployment to self-hosted runner

---

## 🔄 Workflow Stages

### 1. Integration Job

**Purpose**: Validate code before building

**Steps**:
- ✅ Checkout code
- ✅ Set up Python 3.10
- ✅ Install dependencies
- ✅ Lint code structure
- ✅ Verify Dockerfile syntax

**Runs on**: `ubuntu-latest`

---

### 2. Build and Push to ECR

**Purpose**: Build Docker image and push to AWS ECR

**Steps**:
- ✅ Configure AWS credentials
- ✅ Login to Amazon ECR
- ✅ Build Docker image with two tags:
  - `latest` (for easy reference)
  - Commit SHA (for versioning)
- ✅ Push both tags to ECR

**Runs on**: `ubuntu-latest`

**Outputs**:
- `image`: Full image URI
- `image-tag`: Commit SHA tag

---

### 3. Continuous Deployment

**Purpose**: Deploy application to production

**Steps**:
- ✅ Pull latest image from ECR
- ✅ Stop and remove existing container
- ✅ Run new container with proper configuration
- ✅ Verify deployment health
- ✅ Clean up old Docker images

**Runs on**: `self-hosted` (EC2 instance)

---

## 🔧 Key Improvements

### Updated Dependencies
- ✅ Uses latest action versions (`@v4` for checkout, `@v4` for AWS, `@v2` for ECR)
- ✅ Modern GitHub Actions syntax

### Better Error Handling
- ✅ Proper container stop/remove logic
- ✅ Health check verification
- ✅ Deployment verification with retries
- ✅ Clean error messages

### Improved Deployment
- ✅ Container restart policy (`unless-stopped`)
- ✅ Proper environment variables
- ✅ Health check verification (30 attempts, 5s intervals)
- ✅ Automatic cleanup of old images

### Security
- ✅ Uses GitHub secrets for sensitive data
- ✅ Environment-specific deployment
- ✅ Proper credential management

### Image Tagging
- ✅ Both `latest` and commit SHA tags
- ✅ Better version tracking
- ✅ Rollback capability

---

## 📝 Required GitHub Secrets

Configure these secrets in your GitHub repository settings:

```
AWS_ACCESS_KEY_ID          # AWS access key
AWS_SECRET_ACCESS_KEY      # AWS secret key
AWS_REGION                 # AWS region (e.g., us-east-1)
ECR_REPOSITORY_NAME        # ECR repository name
AWS_ECR_LOGIN_URI          # ECR login URI (e.g., 123456789.dkr.ecr.us-east-1.amazonaws.com)
```

### How to Set Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret name and value

---

## 🚀 Workflow Triggers

The workflow runs automatically when:
- ✅ Push to `main` branch
- ❌ Ignores changes to:
  - README.md
  - *.md files
  - Documentation/**
  - .gitignore

---

## 📊 Workflow Outputs

### Build Job Outputs
- `image`: Full ECR image URI with `latest` tag
- `image-tag`: Commit SHA for versioning

### Usage in other jobs:
```yaml
needs: build-and-push-ecr-image
steps:
  - run: echo ${{ needs.build-and-push-ecr-image.outputs.image }}
```

---

## 🐛 Troubleshooting

### Build Fails

**Check**:
1. Dockerfile syntax
2. Requirements.txt dependencies
3. AWS credentials
4. ECR repository exists

### Deployment Fails

**Check**:
1. Self-hosted runner is online
2. Port 8080 is available
3. Docker is running on runner
4. ECR image is accessible

### Health Check Fails

**Check**:
1. Container logs: `docker logs kidney-classifier`
2. Container status: `docker ps`
3. Application logs in container
4. Network connectivity

---

## 🔍 Monitoring

### View Workflow Runs

1. Go to **Actions** tab in GitHub
2. Click on workflow name
3. View logs for each step

### Container Status on Server

```bash
# Check running containers
docker ps

# View container logs
docker logs kidney-classifier -f

# Check container health
docker inspect kidney-classifier | grep -A 10 Health
```

---

## 🔐 Security Best Practices

1. ✅ **Never commit secrets** - Use GitHub Secrets
2. ✅ **Use specific image tags** - Both `latest` and SHA
3. ✅ **Environment protection** - Use GitHub Environments
4. ✅ **Least privilege** - Minimal AWS permissions
5. ✅ **Regular updates** - Keep actions updated

---

## 📈 Performance Optimizations

1. ✅ **Layer caching** - Dockerfile optimized for caching
2. ✅ **Parallel jobs** - Integration and build can be parallel (currently sequential)
3. ✅ **Image cleanup** - Old images cleaned automatically
4. ✅ **Efficient builds** - Only rebuilds on code changes

---

## 🎯 Manual Trigger

You can manually trigger the workflow:

1. Go to **Actions** tab
2. Select workflow
3. Click **Run workflow**
4. Select branch and click **Run workflow**

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [AWS ECR Documentation](https://docs.aws.amazon.com/ecr/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Self-Hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners)

---

## 🔄 Workflow Diagram

```
┌─────────────────┐
│   Push to main  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Integration   │ ◄─── Code validation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Build & Push    │ ◄─── Build Docker image
│      to ECR     │      Push to AWS ECR
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Deployment    │ ◄─── Deploy to EC2
│  (Self-hosted)  │      Health check
└─────────────────┘      Cleanup
```

---

## ✅ Checklist Before Deployment

- [ ] All GitHub secrets are configured
- [ ] ECR repository exists and is accessible
- [ ] Self-hosted runner is configured and online
- [ ] Port 8080 is available on EC2 instance
- [ ] Docker is installed on self-hosted runner
- [ ] AWS IAM user has necessary permissions:
  - [ ] ECR push/pull
  - [ ] EC2 access (if needed)

---

## 🆘 Support

If you encounter issues:

1. Check workflow logs in GitHub Actions
2. Verify secrets are correctly set
3. Check self-hosted runner logs
4. Verify AWS credentials and permissions
5. Check container logs on EC2 instance

---

**Last Updated**: Based on latest GitHub Actions best practices and AWS ECR v2

