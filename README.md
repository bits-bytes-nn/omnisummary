# OmniSummary

AI-powered content summarization agent that processes web articles (HTML), YouTube videos, and PDF documents through Slack, delivering intelligent summaries directly to your channels.

![Summary Preview](./assets/slack.png)

## Overview

OmniSummary is a production-ready agentic system built on [Strands Agents](https://github.com/anthropics/strands) and AWS Bedrock AgentCore. It enables teams to quickly digest technical content by mentioning a Slack bot with a URL, receiving comprehensive summaries with extracted images and reference links.

### Key Features

- **Multi-Format Support**: HTML articles, YouTube videos, PDF documents
- **Intelligent Summarization**: Claude Sonnet 4.5 with customizable opening messages
- **Image Processing**: Automatic extraction and analysis using Claude Haiku 3 VLM
- **Metadata Extraction**: Authors, affiliations, publication dates, keywords, reference URLs
- **Dual Slack Workspace**: Support for both personal and business channels
- **Production Infrastructure**: AWS Bedrock AgentCore with event deduplication and signature verification

### Use Cases

- Research teams tracking arxiv papers and technical blogs
- Engineering teams digesting conference talks and tutorials
- Product teams reviewing industry reports and whitepapers
- Knowledge sharing across personal and business Slack workspaces

## Architecture

```
┌─────────────┐
│ Slack User  │ @your-bot https://example.com/article
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│ API Gateway → Lambda (Slack Event Handler)      │
│ - Signature verification                        │
│ - Event deduplication (DynamoDB)                │
│ - Immediate ACK                                 │
└──────┬──────────────────────────────────────────┘
       │ (async invoke)
       ▼
┌─────────────────────────────────────────────────┐
│ AWS Bedrock AgentCore Runtime                   │
│ - Docker container (ECS-like)                   │
│ - Strands Agent orchestration                   │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│ Agent Tools                                     │
│ 1. parse_html / parse_pdf / parse_youtube       │
│ 2. generate_summary                             │
│ 3. send_slack_message                           │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│ Slack Channels                                  │
│ - Summary with images                           │
│ - Reference URLs                                │
└─────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | Strands Agents |
| Runtime Platform | AWS Bedrock AgentCore |
| Infrastructure | AWS CDK (Python) |
| LLM | Amazon Bedrock (Claude Sonnet 4.5) |
| VLM | Amazon Bedrock (Claude Haiku 3) |
| PDF Parsing | Unstructured / Upstage Document Parser |
| Messaging | Slack Bot API |
| Package Manager | uv |

## Prerequisites

Before starting, ensure you have:

- **AWS Account** with Administrator access
- **AWS CLI** configured with credentials
- **Slack Workspace** with Admin permissions to create apps
- **Python 3.12+** installed
- **uv** package manager ([installation guide](https://github.com/astral-sh/uv))
- **Docker** for building AgentCore runtime images

## Installation

### 1. Local Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd omnisummary

# Install dependencies using uv
uv sync

# Create environment configuration
cp .env.template .env
# Edit .env with your credentials (see below)

# Create config file
cp config/config-template.yaml config/config.yaml
# Edit config.yaml (see Configuration section)
```

### 2. Slack App Setup

This section provides detailed instructions for creating and configuring your Slack App.

#### 2.1 Create Slack App

1. Go to [Slack API](https://api.slack.com/apps)
2. Click **"Create New App"** → **"From scratch"**
3. Enter:
   - **App Name**: Choose any name (e.g., "ContentSummary", "DigestBot", "OmniSummary")
   - **Workspace**: Select your workspace
4. Click **"Create App"**

> **Note**: Throughout this README, we use `@omnisummary` as an example. Replace it with your actual bot name.

#### 2.2 Configure OAuth & Permissions

1. Navigate to **"OAuth & Permissions"** in the sidebar
2. Scroll to **"Scopes"** → **"Bot Token Scopes"**
3. Add the following scopes:
   ```
   app_mentions:read    - Detect when bot is mentioned
   chat:write           - Send messages to channels
   files:write          - Upload images (PDF thumbnails)
   channels:history     - (Optional) Read channel history
   ```
4. Scroll to top and click **"Install to Workspace"**
5. Authorize the app
6. **Copy the Bot User OAuth Token** (starts with `xoxb-`)
   - Save this as `SLACK_PERSONAL_TOKEN` or `SLACK_BUSINESS_TOKEN` in `.env`

#### 2.3 Enable Event Subscriptions

**Important**: Complete this step AFTER deploying the AWS infrastructure (Step 3).

1. Navigate to **"Event Subscriptions"**
2. Toggle **"Enable Events"** to ON
3. Enter **Request URL**:
   ```
   https://<your-api-gateway-url>/slack/events
   ```
   You'll get the API Gateway URL from AWS deployment outputs.
4. Under **"Subscribe to bot events"**, add:
   ```
   app_mention
   ```
5. Click **"Save Changes"**

#### 2.4 Get Channel IDs

Channel IDs are required in your configuration to specify where summaries should be posted.

**Method 1: Slack Web/Desktop App**
1. Right-click the channel name
2. Select **"View channel details"**
3. Scroll down and copy the **Channel ID** (starts with `C`)

**Method 2: Slack API**
```bash
curl -H "Authorization: Bearer xoxb-YOUR-TOKEN" \
  https://slack.com/api/conversations.list
```

**Multiple Channels**: Separate with commas (no spaces)
```
SLACK_PERSONAL_CHANNEL_IDS=C123ABC456,C789DEF012,C345GHI678
```

#### 2.5 Get Signing Secret

1. Navigate to **"Basic Information"**
2. Scroll to **"App Credentials"**
3. Copy the **Signing Secret**
4. Save as `SLACK_SIGNING_SECRET` in `.env`

#### 2.6 Invite Bot to Channels

After app installation, invite your bot to channels:

```
/invite @your-bot-name
```

The bot must be a member of all channels listed in `SLACK_PERSONAL_CHANNEL_IDS` and `SLACK_BUSINESS_CHANNEL_IDS`.

### 3. Environment Variables Configuration

Edit `.env` with your credentials:

```bash
# Slack Configuration
SLACK_PERSONAL_TOKEN=xoxb-your-personal-workspace-token
SLACK_PERSONAL_CHANNEL_IDS=C123ABC456,C789DEF012
SLACK_BUSINESS_TOKEN=xoxb-your-business-workspace-token  # Optional
SLACK_BUSINESS_CHANNEL_IDS=C345GHI678                    # Optional
SLACK_SIGNING_SECRET=your-signing-secret

# Optional: Upstage Document Parser (for PDF parsing)
UPSTAGE_API_KEY=your-upstage-api-key

# Optional: Langchain Tracing
LANGCHAIN_API_KEY=your-langchain-api-key
LANGCHAIN_TRACING_V2=false
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=omnisummary

# Logging
LOG_LEVEL=INFO
```

**Slack Configuration Notes:**

- **Personal vs Business Tokens**: You can configure separate Slack workspaces
  - `SLACK_PERSONAL_*`: Always used for summaries
  - `SLACK_BUSINESS_*`: Only used when explicitly requested (e.g., "send to business channels")
- **Channel IDs**: Comma-separated, no spaces
- **Signing Secret**: Required for request verification and security

### 4. Config File Configuration

Edit `config/config.yaml`:

```yaml
resources:
  project_name: omnisummary
  stage: dev  # or prod
  profile_name: your-aws-profile  # AWS CLI profile name
  default_region_name: us-west-2
  bedrock_region_name: us-west-2  # Must support Claude models

  # Optional: Use existing VPC
  vpc_id: vpc-xxxxx
  subnet_ids:
    - subnet-xxxxx
    - subnet-yyyyy

  # Enable business Slack workspace
  enable_business_slack_channels: true  # Set to false to disable

content_parser:
  figure_analysis_model_id: anthropic.claude-3-haiku-20240307-v1:0
  metadata_extraction_model_id: anthropic.claude-sonnet-4-5-20250929-v1:0
  pdf_parser_type: unstructured  # or upstage

summarization:
  summarization_model_id: anthropic.claude-sonnet-4-5-20250929-v1:0
  n_thumbnails: 1  # Number of images to extract

agent:
  agent_model_id: anthropic.claude-sonnet-4-5-20250929-v1:0

infrastructure:
  lambda_timeout_seconds: 60
  lambda_memory_mb: 128
  agentcore_image_tag: latest
```

**Important Configuration Notes:**

- `bedrock_region_name`: Must be a region where Claude Sonnet 4.5 is available (e.g., `us-west-2`, `us-east-1`)
- `enable_business_slack_channels`: Must be `true` for business workspace features to work
- `pdf_parser_type`: `unstructured` (free, local) or `upstage` (API-based, requires key)

### 5. AWS Deployment

#### 5.1 CDK Bootstrap (First-time only)

```bash
# Export AWS profile if using named profile
export AWS_PROFILE=your-aws-profile

# Bootstrap CDK (creates S3 bucket and IAM roles)
uv run cdk bootstrap
```

#### 5.2 Deploy Foundation Stack

```bash
# Deploys: VPC, ECR, DynamoDB, IAM Roles, CodeBuild
uv run cdk deploy OmniSummaryFoundationStack

# Review changes and confirm with 'y'
```

This stack creates:
- ECR repository for AgentCore Docker images
- DynamoDB table for event deduplication
- CodeBuild project for automatic Docker builds
- IAM roles for Lambda and AgentCore

#### 5.3 Build and Push Docker Image

After Foundation Stack deployment, the CodeBuild project automatically triggers to build and push the Docker image to ECR. You can monitor the build:

```bash
# Get build project name from CloudFormation outputs
aws codebuild list-builds-for-project \
  --project-name omnisummary-dev-summary-bot-builder

# Check build status
aws codebuild batch-get-builds --ids <build-id>
```

**Manual build (if needed):**
```bash
# Get ECR repository URI from CloudFormation outputs
aws ecr describe-repositories --repository-names omnisummary-dev-summary-bot

# Login to ECR
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com

# Build and push
docker build -t omnisummary-dev-summary-bot:latest .
docker tag omnisummary-dev-summary-bot:latest <ecr-uri>:latest
docker push <ecr-uri>:latest
```

#### 5.4 Deploy Application Stack

```bash
# Deploys: AgentCore Runtime, Lambda, API Gateway, SSM Parameters
uv run cdk deploy OmniSummaryApplicationStack

# Save the API Gateway URL from outputs
```

**Important**: Copy the API Gateway URL from deployment outputs. You'll need it for Slack Event Subscriptions (Step 2.3).

Example output:
```
OmniSummaryApplicationStack.ApiEndpoint = https://abcdef1234.execute-api.us-west-2.amazonaws.com/dev
```

#### 5.5 Update Slack Event Subscription URL

Now go back to Step 2.3 and configure the Event Subscription URL using the API Gateway endpoint:
```
https://abcdef1234.execute-api.us-west-2.amazonaws.com/dev/slack/events
```

#### 5.6 Verify Deployment

```bash
# Check AgentCore Runtime status
aws bedrock-agentcore list-agent-runtimes

# Check Lambda function
aws lambda list-functions --query "Functions[?contains(FunctionName, 'slack-event-handler')]"

# Test API Gateway
curl -X POST https://<your-api-gateway-url>/slack/events
# Should return 401 (Unauthorized) without valid Slack signature
```

## Usage

### Basic Commands

Once deployed and configured, interact with your bot in Slack:

#### Simple Summarization

```
@your-bot-name https://example.com/blog-post
```

The bot will:
1. Parse the HTML content
2. Extract metadata (title, authors, date, keywords)
3. Generate a summary
4. Post to personal Slack channels

#### YouTube Videos

```
@your-bot-name https://youtube.com/watch?v=abcd1234
```

Extracts transcript and summarizes video content.

#### PDF Documents

```
@your-bot-name https://arxiv.org/pdf/2301.12345.pdf
```

Parses PDF, extracts figures, and generates summary with image thumbnails.

### Advanced Features

#### Send to Business Channels

Add keywords to send summaries to both personal AND business workspaces:

```
@your-bot-name https://example.com/article 회사 채널에도 보내줘
@your-bot-name https://example.com/article send to business channels
@your-bot-name https://example.com/article 전체 채널에
```

Keywords detected:
- Korean: `회사 채널`, `비즈니스 채널`, `업무 채널`, `전체 채널`, `다 보내`
- English: `business channel`, `work channel`, `send to all`

**Requirement**: `enable_business_slack_channels: true` in `config.yaml`

#### Custom Opening Message

Start the summary with a specific phrase:

```
@your-bot-name https://example.com/article AWS 신기능이네요로 시작해줘
```

The summary will begin with: "AWS 신기능이네요"

Keywords: `로 시작해줘`, `로 시작`, `부터 시작`

#### Combined Features

```
@your-bot-name https://example.com/article
흥미로운 아키텍처네요로 시작. 회사 채널에도 공유해줘
```

This will:
1. Generate summary starting with "흥미로운 아키텍처네요"
2. Post to both personal and business channels

### Output Format

Summaries include:

```
🗞️ 2024-01-15에 John Doe, Jane Smith에서 발행한 Article Title의 요약입니다.

[Opening commentary in Gruber-style voice]

📌 왜 여기에 주목해야 하나요?
[Significance and context]

🔄 핵심 아이디어 및 접근 방식
[Key concepts and methodologies]

🛠️ 기술적 심층 분석
[Technical details and implementation]

📊 성과 및 비즈니스 임팩트
[Results and metrics]

📎 참고 링크
- https://github.com/example/repo
- https://arxiv.org/abs/2301.12345
```

For PDFs, image thumbnails are uploaded separately after the text summary.

## Project Structure

```
omnisummary/
├── agent/                      # Strands Agent definition
│   ├── agent.py               # summarization_agent
│   ├── agent_tools.py         # Tool implementations
│   └── tool_state.py          # Stateful tool execution
├── agent_runtime/             # Bedrock AgentCore entrypoint
│   └── app.py                 # BedrockAgentCoreApp
├── config/                    # Configuration files
│   ├── config-template.yaml
│   └── config.yaml            # User config (gitignored)
├── infrastructure/            # AWS CDK stacks
│   ├── foundation_stack.py    # VPC, ECR, DynamoDB, IAM
│   └── application_stack.py   # AgentCore, Lambda, API Gateway
├── lambda_handlers/           # Lambda functions
│   └── slack_event_handler.py # Slack event processing
├── shared/                    # Shared utilities
│   ├── config.py              # Config models
│   ├── constants.py           # Enums and constants
│   ├── formatters.py          # Slack message formatting
│   ├── models.py              # Data models (Pydantic)
│   └── prompts/               # LLM prompts
│       └── prompts.py
├── tools/                     # Content processing tools
│   ├── content_parser/        # HTML, PDF, YouTube parsers
│   ├── output_handler/        # Slack message sender
│   └── summarizer/            # LLM summarization
├── tests/                     # Unit and integration tests
├── .env.template              # Environment variable template
├── config.yaml                # Main configuration
├── pyproject.toml             # Python project definition
├── Dockerfile                 # AgentCore runtime image
└── README.md                  # This file
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=omnisummary --cov-report=html

# Run specific test file
uv run pytest tests/test_agents.py
```

### Code Quality

```bash
# Format code
uv run black .

# Lint
uv run ruff check .

# Type checking
uv run mypy .
```

### Local Testing (Without AWS)

You can test agent logic locally:

```python
from agent import summarization_agent

# Test the agent
response = summarization_agent("https://example.com/article")
print(response)
```

**Note**: This requires configuring local `.env` with Slack tokens and setting up AWS credentials for Bedrock access.

### Adding New Tools

1. Define tool function in `agent/agent_tools.py`:
   ```python
   @tool
   async def my_new_tool(param: str) -> str:
       """Tool description for LLM."""
       # Implementation
       return result
   ```

2. Add tool to agent in `agent/agent.py`:
   ```python
   tools = [
       generate_summary,
       parse_html,
       parse_pdf,
       parse_youtube,
       send_slack_message,
       my_new_tool,  # Add here
   ]
   ```

3. Update system prompt to instruct agent when to use the tool

4. Rebuild and redeploy:
   ```bash
   # Trigger CodeBuild
   aws codebuild start-build --project-name omnisummary-dev-summary-bot-builder

   # After build completes, update AgentCore runtime
   uv run cdk deploy OmniSummaryApplicationStack
   ```

## Troubleshooting

### Bot doesn't respond to mentions

**Check 1: Bot is invited to channel**
```
/invite @your-bot-name
```

**Check 2: Event Subscription URL is verified**
- Go to Slack App → Event Subscriptions
- URL should show green checkmark

**Check 3: Check Lambda logs**
```bash
aws logs tail /aws/lambda/omnisummary-dev-slack-event-handler-lambda --follow
```

**Check 4: Verify app_mention scope**
- Slack App → OAuth & Permissions → Scopes
- Ensure `app_mentions:read` is added

### Summaries not appearing

**Check 1: AgentCore Runtime logs**
```bash
# Get runtime name
aws bedrock-agentcore list-agent-runtimes

# Get log group
aws logs describe-log-groups --query "logGroups[?contains(logGroupName, 'agentcore')]"

# Tail logs
aws logs tail <log-group-name> --follow
```

**Check 2: Slack tokens are correct**
```bash
# Verify token works
curl -H "Authorization: Bearer xoxb-YOUR-TOKEN" \
  https://slack.com/api/auth.test
```

**Check 3: Channel IDs are correct**
```bash
# List channels the bot is member of
curl -H "Authorization: Bearer xoxb-YOUR-TOKEN" \
  https://slack.com/api/conversations.list
```

### Business channels not receiving messages

**Check 1: Config setting**
```yaml
# config/config.yaml
resources:
  enable_business_slack_channels: true  # Must be true
```

**Check 2: Mention business keywords**
```
@bot https://example.com/article 회사 채널에도 보내줘
```

**Check 3: Business token and channel IDs are set**
```bash
# .env
SLACK_BUSINESS_TOKEN=xoxb-...
SLACK_BUSINESS_CHANNEL_IDS=C123ABC456
```

### Docker build fails

**Check 1: ECR authentication**
```bash
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com
```

**Check 2: CodeBuild logs**
```bash
aws codebuild list-builds-for-project --project-name omnisummary-dev-summary-bot-builder
aws codebuild batch-get-builds --ids <build-id>
```

### DynamoDB event deduplication issues

If you're seeing duplicate events:

```bash
# Check DynamoDB table
aws dynamodb scan --table-name omnisummary-dev-event-deduplication

# Verify TTL is enabled
aws dynamodb describe-time-to-live --table-name omnisummary-dev-event-deduplication
```

## Cost Estimation

Approximate monthly costs for moderate usage (100 summaries/day):

| Service | Usage | Cost |
|---------|-------|------|
| Bedrock Claude Sonnet 4.5 | ~50K tokens/summary × 100 | ~$15 |
| Bedrock Claude Haiku 3 | ~10K tokens/image × 100 | ~$1 |
| Lambda | 100 invocations/day | ~$0.01 |
| API Gateway | 100 requests/day | ~$0.01 |
| DynamoDB | On-demand, minimal | ~$0.25 |
| AgentCore Runtime | Runtime hours | Variable* |
| ECR | 10 images stored | ~$0.10 |
| **Total** | | **~$16-20/month** |

*AgentCore Runtime costs depend on task duration and concurrency. Idle time may incur minimal charges.

**Cost Optimization Tips:**
- Use `dev` stage for testing (enables aggressive cleanup)
- Set `n_thumbnails: 1` to minimize VLM API calls
- Configure lifecycle rules on ECR (already included)
- Use Bedrock on-demand pricing for variable workloads

## Security Best Practices

1. **Slack Request Verification**: Always enabled via signing secret
2. **Secrets Management**: All credentials stored in SSM Parameter Store (encrypted)
3. **Event Deduplication**: Prevents replay attacks and duplicate processing
4. **IAM Least Privilege**: Separate roles for Lambda and AgentCore with minimal permissions
5. **VPC Isolation**: Optional VPC deployment for network isolation
6. **Environment Separation**: Use `dev`/`prod` stages with separate AWS accounts

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow code standards in `CLAUDE.md`
4. Write tests for new functionality
5. Run code quality checks (`black`, `ruff`, `mypy`)
6. Commit with descriptive messages
7. Push and create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [Strands Agents](https://github.com/anthropics/strands) by Anthropic
- Uses AWS Bedrock AgentCore for production deployment
- Inspired by Daring Fireball's analytical writing style
- PDF parsing powered by Unstructured and Upstage

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review CloudWatch Logs for debugging

---

**Note**: Replace `@your-bot-name` with your actual Slack bot display name throughout usage examples.
