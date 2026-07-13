# Building Multi-Agent Social Intelligence with Strands Agents and Amazon Bedrock AgentCore

A multi-agent system built with [Strands Agents SDK](https://github.com/strands-agents/sdk-python) on [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/). It discovers qualified prospects, collects multi-signal trend data, scores relevance, and generates personalized outreach emails.

## Architecture Overview

![Architecture overview: a browser SPA invokes the Amazon Bedrock AgentCore Runtime, which runs Strands agents against Amazon Bedrock, AgentCore Memory, and Observability, calls tools through the AgentCore Gateway to an AWS Lambda handler over nine external APIs, and persists leads to Amazon DynamoDB.](architecture/architecture-overview.png)

The front-end application invokes the Amazon Bedrock AgentCore Runtime, which hosts four Strands agents under either the Swarm or Graph orchestration pattern. Agents call Claude Sonnet 4.6 on Amazon Bedrock, keep session context in AgentCore Memory, and emit traces to AgentCore Observability. Tool calls travel over MCP through the AgentCore Gateway (IAM auth) to a single AWS Lambda handler that fans out to nine external APIs. Scored leads and email drafts persist to Amazon DynamoDB.

Deploy once, then test both orchestration patterns through the same endpoint. Diagram sources live in [`architecture/`](architecture/) as editable draw.io files.

## Key Features

- **4 specialized agents** with focused system prompts and Pydantic structured output
- **9 API tools** behind a single Lambda: Hacker News, YouTube, dev.to, Wikipedia, GitHub, Lobste.rs, Product Hunt, Reddit, Stack Overflow
- **AgentCore Gateway** with IAM auth for MCP-based tool discovery (CDK L2 constructs)
- **AgentCore Runtime** for managed agent deployment (CDK L2 constructs)
- **Tool schema** as single source of truth for tool contracts
- **2 orchestration patterns**: Graph (deterministic DAG) and Swarm (autonomous handoffs), both with orchestrator-level Memory
- **DRY tool architecture**: shared handlers behind a single Lambda
- **AgentCore Memory**: created by the CDK stack, with short-term session context plus long-term summarization and semantic recall. Graceful degradation when unset.
- **Optional model tiering**: per-agent model overrides (`TREND_MODEL_ID`, `SEARCH_MODEL_ID`, `ANALYSIS_MODEL_ID`, `EMAIL_MODEL_ID`) default to the documented Claude Sonnet 4.6
- **Optional Bedrock Guardrails**: set `GUARDRAIL_ID` and `GUARDRAIL_VERSION` to enable prompt-injection filtering and content controls on every agent call
- **Frontier dedup table**: cross-agent DynamoDB table prevents the same prospect from being scored and emailed twice across sessions
- **Optional grounding gate**: set `GROUNDING_MIN_SCORE` to require a minimum factual-grounding score before the agent sends an email (`src/social_intelligence/tools/grounding_gate.py`)
- **Optional compliance footer**: set `COMPLIANCE_FOOTER_REQUIRED=true` to append a regulatory opt-out footer to every generated email
- **Optional human review**: set `EMAIL_APPROVAL_REQUIRED=true` to route emails through a human-approval step before delivery
- **Quality eval harness**: `scripts/eval_quality.py` compares agent output against `eval/golden_set.json` reference answers and reports precision, recall, and scoring accuracy
- **AWS CDK infrastructure** with cdk-nag security compliance

## Project Structure

```
.
├── src/
│   └── social_intelligence/           # Main package (src layout)
│       ├── __init__.py
│       ├── config.py                      # Shared config (model ID, region)
│       ├── agents/
│       │   ├── trend_research_agent.py    # Prospect discovery + trend signals
│       │   ├── search_specialist_agent.py # Enrichment + competitive intel
│       │   ├── analysis_agent.py          # Scoring (0-100) + prioritization
│       │   └── email_generation_agent.py  # Personalized outreach + lead persistence
│       ├── tools/
│       │   ├── registry.py                # Tool route registry (add new tools here)
│       │   ├── _secrets.py                # Shared Secrets Manager helper
│       │   ├── _freshness.py              # Temporal decay weight helper
│       │   ├── hackernews.py              # Hacker News Firebase API
│       │   ├── youtube.py                 # YouTube Data API v3
│       │   ├── devto.py                   # dev.to (Forem) API
│       │   ├── _http.py                   # Shared HTTP retry helper
│       │   ├── wikipedia.py               # Wikipedia REST API
│       │   ├── github.py                  # GitHub Search API
│       │   ├── lobsters.py                # Lobste.rs JSON API
│       │   ├── producthunt.py             # Product Hunt GraphQL API
│       │   ├── reddit.py                  # Reddit JSON API (intent signal detection)
│       │   ├── stackoverflow.py           # Stack Exchange API (demand signals)
│       │   ├── brand_knowledge.py         # Brand KB (agent-side)
│       │   ├── dynamodb_tool.py           # Lead persistence + dedup (agent-side)
│       │   ├── grounding_gate.py          # Optional grounding check before email send
│       │   └── email_renderer.py          # Jinja2 HTML email rendering (agent-side)
│       ├── orchestration/
│       │   ├── graph_runner.py            # CLI: invoke deployed agent (graph pattern)
│       │   └── swarm_runner.py            # CLI: invoke deployed agent (swarm pattern)
│       └── schemas/
│           ├── models.py                  # Pydantic inter-agent data contracts
│           ├── tool_schema.json           # Tool API contract (single source of truth)
│           └── openapi.yaml               # OpenAPI spec generated from tool schema
├── tests/                                 # Test suite
│   ├── test_tools.py                      # Unit tests for tool handlers
│   ├── test_hardening.py                  # Security hardening tests
│   ├── test_coverage.py                   # Coverage tests for previously-untested paths
│   └── integration/
│       └── test_runtime.py                # End-to-end AgentCore Runtime test
├── entrypoint.py                          # AgentCore Runtime entrypoint
├── pyproject.toml                         # Modern Python packaging
├── demo/
│   ├── app.py                             # Streamlit demo UI (main entry)
│   ├── config.py                          # Configuration, env loading, constants
│   ├── streaming.py                       # AgentCore streaming client + SSE parser
│   ├── components.py                      # Reusable UI rendering components
│   └── data.py                            # DynamoDB data layer + config checks
├── infra/
│   ├── app.py                             # CDK app entry point
│   ├── gateway_stack.py                   # CDK stack: AgentCore Gateway + IAM target
│   ├── cdk.json
│   ├── requirements.txt
│   ├── stacks/
│   │   └── social_intelligence_stack.py   # CDK stack (Lambda + Gateway + Runtime)
│   └── lambda/
│       ├── handler.py                     # Lambda router (imports from tools/)
│       └── requirements.txt
├── scripts/
│   ├── benchmark.py                       # Load and latency benchmarks
│   ├── eval_quality.py                    # Quality evaluation harness (golden set)
│   └── test_invoke.py                     # Quick smoke test for deployed runtime
├── config/
│   └── icp_profile.json                   # Ideal customer profile configuration
├── eval/
│   └── golden_set.json                    # Reference outputs for quality evaluation
├── .env.example                           # Environment variable template
├── LICENSE
├── CONTRIBUTING.md
└── CODE_OF_CONDUCT.md
```

## Adding a New Tool

The architecture follows the Open/Closed Principle; adding a tool requires no changes to existing code:

1. Create `src/social_intelligence/tools/<name>.py` with a `handle(params: dict) -> dict` function
2. Add one line to `src/social_intelligence/tools/registry.py`: `"<name>": <module>.handle`
3. Add the tool definition to `src/social_intelligence/schemas/tool_schema.json`

The Lambda handler and AgentCore Gateway pick up the new tool automatically.

## Prerequisites

- Python 3.12+
- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- [AWS CDK v2](https://docs.aws.amazon.com/cdk/v2/guide/getting-started.html) (`npm install -g aws-cdk@2.200.0`)
- [Amazon Bedrock AgentCore CLI](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore-getting-started.html)
- Access to Amazon Bedrock foundation models (Anthropic Claude)
- **Supported regions:** US East (N. Virginia) `us-east-1`, US West (Oregon) `us-west-2`, Europe (Ireland) `eu-west-1` (requires Amazon Bedrock AgentCore availability)

> **Cost warning:** You are responsible for the cost of the AWS services used while running this sample. See the [Cost](#cost) section for estimates before deploying.

## Getting Started

> **Estimated time:** ~45–60 minutes (including CDK deployment)

### 1. Install dependencies

```bash
git clone https://github.com/aws-samples/sample-multi-agent-social-intelligence-strands-agentcore.git
cd sample-multi-agent-social-intelligence-strands-agentcore
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Deploy infrastructure

```bash
cd infra && pip install -r requirements.txt

export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export CDK_DEFAULT_REGION=us-east-1

cdk bootstrap && cdk deploy
```

CDK deploys everything in a single stack: the Lambda, API Gateway, AgentCore Gateway (IAM auth + Lambda target), AgentCore Runtime, AgentCore Memory, DynamoDB tables, a Bedrock Guardrail, and four CMK-encrypted Secrets Manager secrets. No separate gateway setup script is needed.

### 3. Populate API credentials (optional)

The CDK stack creates the secrets empty and CMK-encrypted. The sample runs with zero credentials (the tools fall back to unauthenticated requests), so this step is optional. To enable authenticated calls, set values WITHOUT routing them through CloudFormation or git:

```bash
# YouTube Data API key (raises YouTube tool reliability)
aws secretsmanager put-secret-value \
  --secret-id social-intel/youtube-api-key --secret-string "YOUR_YOUTUBE_API_KEY"

# Optional: GitHub token (raises GitHub rate limit 60 -> 5000 req/hr)
aws secretsmanager put-secret-value \
  --secret-id social-intel/github-token --secret-string "YOUR_GITHUB_PAT"

# Optional: Reddit OAuth2 client credentials (JSON)
aws secretsmanager put-secret-value \
  --secret-id social-intel/reddit-oauth \
  --secret-string '{"client_id":"...","client_secret":"..."}'

# Optional: Product Hunt API token
aws secretsmanager put-secret-value \
  --secret-id social-intel/producthunt-api-token --secret-string "YOUR_PH_TOKEN"
```

The exact secret names are in the `ApiSecretNames` CDK stack output.

### 4. Invoke

```bash
# Graph pattern (default)
agentcore invoke '{"prompt": "Find recent AI tool launches and generate outreach emails"}'

# Swarm pattern
agentcore invoke '{"prompt": "Deep-dive on AI agent frameworks", "pattern": "swarm"}'

# With memory
agentcore invoke '{"prompt": "...", "session_id": "session-001", "actor_id": "analyst-1"}'

# Via orchestration runners (compare patterns side by side)
export AGENTCORE_AGENT_ARN=<RuntimeArn from CDK output>
python -m social_intelligence.orchestration.graph_runner "Find recent AI tool launches"
python -m social_intelligence.orchestration.swarm_runner "Deep-dive on AI agent frameworks"
```

### 5. Run the Streamlit demo (optional)

```bash
pip install -e ".[demo]"
cp .env.example .env
# Edit .env: set AGENTCORE_AGENT_ARN and AWS_DEFAULT_REGION from CDK outputs
streamlit run demo/app.py
```

The demo connects to the deployed AgentCore Runtime and streams agent events in real time. It reads leads from DynamoDB and renders HTML email previews.

### 6. Create the DynamoDB table

The CDK stack creates the `social-intel-leads` table automatically. If you need to create it manually:

```bash
aws dynamodb create-table \
  --table-name social-intel-leads \
  --attribute-definitions \
    AttributeName=prospect_id,AttributeType=S \
    AttributeName=discovered_at,AttributeType=S \
  --key-schema \
    AttributeName=prospect_id,KeyType=HASH \
    AttributeName=discovered_at,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

## Orchestration Patterns: Graph vs Swarm

| | Graph (default) | Swarm |
|---|---|---|
| Control flow | Deterministic DAG | Autonomous handoffs |
| Execution order | Predictable: research + search → analysis → email | Dynamic: agents self-organize |
| Parallelism | Built-in: research and search run concurrently | Sequential handoffs |
| Conditional logic | Explicit edge conditions (score ≥ 60 gates email) | Agents decide |
| Structured output | Enabled (Pydantic models enforce contracts) | Disabled (would block handoffs) |
| Token efficiency | Higher (skips email for low-score prospects) | Lower (agents may revisit work) |
| Safety | DAG prevents cycles | Ping-pong detection (window=8, min 3 unique) |
| Best for | Production pipelines with known workflows | Exploratory, open-ended research |

**Implementation note:** In Swarm mode, `structured_output_model` is disabled on individual agents because Strands treats structured output as a task-completion signal. With it enabled, an agent would stop instead of handing off to the next specialist. Data validation happens at the orchestration boundary instead. Graph mode keeps `structured_output_model` enabled because each node runs to completion before the next one starts.

Both patterns run the same four agents; the diagrams below show only the flow topology, and the tables that follow give the per-agent detail.

### Agents, tools, and outputs

| Agent | node_id | Tools | Output |
|---|---|---|---|
| Trend Research | `research` | `hackernews_trending`, `youtube_trending`, `devto_trending`, `producthunt_trending`, `reddit_search`, `check_existing_leads` | `TrendData` |
| Search Specialist | `search` | `wikipedia_summary`, `github_search`, `lobsters_trending`, `stackoverflow_search`, `check_existing_leads` | `EnrichmentData` |
| Analysis | `analysis` | none (reasoning only) | `ScoredProspect` |
| Email Generation | `email` | `retrieve_brand_knowledge`, `render_email_html`, `verify_email_claims`, `store_lead` | `EmailDraft` |

The Analysis agent scores each prospect 0–100 on Claude Sonnet 4.6, then applies modifiers (temporal decay 1.5x/0.5x, ICP match ±10):

| Scoring dimension | Max points |
|---|---|
| Topical alignment | 25 |
| Timing relevance | 20 |
| Engagement potential | 20 |
| Intent signals | 20 |
| Data quality | 15 |

### Graph pattern

![Graph pattern: user request fans out to the research and search agents in parallel, a wait-for-both gate feeds the analysis agent, and a score >= 60 condition gates the email agent, which persists to Amazon DynamoDB.](architecture/graph-pattern.png)

The Graph pattern runs a deterministic DAG. Research and search execute in parallel, `_all_dependencies_complete()` gates the analysis node, and `_score_above_threshold()` (default `EMAIL_SCORE_THRESHOLD=60`) decides whether the email node runs. Low-score prospects skip email generation and save tokens. Graph configuration: `set_execution_timeout(1200)`, `set_max_node_executions(20)`, `set_node_timeout(300)`.

### Swarm pattern

![Swarm pattern: four agents share a working memory and hand off dynamically, starting from the trend research entry point, with optional re-query and loopback edges, bounded by swarm safety mechanisms.](architecture/swarm-pattern.png)

The Swarm pattern gives every agent the full task context and lets each one choose its own handoff target via `handoff_to_agent()`. The sequence emerges from reasoning rather than a fixed graph. Safety knobs (`max_handoffs=15`, `max_iterations=15`, repetitive-handoff detection over a window of 8 with 3 minimum unique agents) bound the loop.

## Tool Architecture

```
src/social_intelligence/schemas/tool_schema.json  ← Single source of truth (tool contracts)
        │
        ├──→ AgentCore Gateway    (discovers tools via tool schema)
        ├──→ API Gateway          (request validation)
        └──→ Documentation        (auto-generated)

src/social_intelligence/tools/<name>.py        ← Shared business logic (pure functions)
        │
        └──→ infra/lambda/handler.py   (Lambda router via tools/registry.py)
```

## Cost

You are responsible for the cost of the AWS services used while running this solution. As of June 2026, processing 50 prospects costs approximately $3 to $5 in the US East (N. Virginia) Region using Amazon Bedrock on-demand pricing for Claude Sonnet 4 (based on testing with default prompt caching configuration, June 2026). Actual costs vary with prompt caching configuration and the number of tool calls per run.

We recommend creating a budget through [AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/) to monitor spending.

## Cleanup

The stack uses an env-gated removal policy. Stateful resources (DynamoDB tables, KMS key, Secrets Manager secrets) are RETAINED by default and DESTROYED only when `CDK_ENV=dev`. For a full teardown of a throwaway environment:

```bash
cd infra
CDK_ENV=dev cdk destroy --all
```

This removes the Lambda, API Gateway, AgentCore Gateway/Runtime/Memory, DynamoDB tables, the Bedrock Guardrail, the KMS key, and the four Secrets Manager secrets the stack created. Without `CDK_ENV=dev`, the secrets, tables, and key are retained; delete them manually if you want them gone:

```bash
for s in youtube-api-key producthunt-api-token github-token reddit-oauth; do
  aws secretsmanager delete-secret --secret-id "social-intel/$s" --force-delete-without-recovery
done
```

**Verify cleanup:**

```bash
# Confirm stack deletion
aws cloudformation describe-stacks --stack-name SocialIntelligenceStack --region us-east-1
# Should return: "Stack not found"

# Check for ongoing charges
# Visit AWS Cost Explorer: https://aws.amazon.com/aws-cost-management/aws-cost-explorer/
```

## Troubleshooting

| Issue | Likely Cause | Quick Fix |
|---|---|---|
| `ModuleNotFoundError: social_intelligence` | Package not installed in editable mode | Run `pip install -e .` from project root |
| Gateway returns 403 Forbidden | IAM auth not configured for the caller | Verify `aws sts get-caller-identity` matches the account. Re-run `cdk deploy`. |
| `bedrock-agentcore` import fails | Missing optional dependency | Run `pip install "bedrock-agentcore[strands-agents]"` |
| YouTube tool returns empty results | Missing or expired API key | Check `aws secretsmanager get-secret-value --secret-id social-intel/youtube-api-key` |
| Streamlit shows "No leads yet" | DynamoDB table missing or pipeline not run | Deploy CDK stack (creates table), then run a pipeline |
| Streamlit shows config errors | Missing `.env` file | Copy `.env.example` to `.env` and fill in `AGENTCORE_AGENT_ARN` |
| `store_lead` returns ResourceNotFoundException | DynamoDB table not in the deployment region | Redeploy CDK stack or create table manually (see Getting Started) |

## Responsible Use

This sample demonstrates AI-powered prospect discovery and outreach using publicly
available data. When adapting this system:

- Only collect and process publicly available information
- Comply with applicable data protection regulations (GDPR, CCPA, etc.)
- Include opt-out mechanisms in all outreach communications
- Do not use this system for surveillance, social scoring, or profiling
- Review generated emails before sending; do not automate sending without human oversight
- Respect platform terms of service when collecting data from third-party APIs

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for guidelines on reporting bugs, submitting pull requests, and adding new tools.

## Security

- IAM authentication on all endpoints (AgentCore Gateway, API Gateway)
- Secrets Manager for API keys (never hardcoded)
- Least-privilege IAM with specific resource ARNs
- cdk-nag AWS Solutions checks (0 findings)
- CloudWatch logging for Lambda and API Gateway
- Input validation and sanitization in all tool handlers (length limits, allowed values, regex filtering). User prompts are passed to Amazon Bedrock models via parameterized API calls.
- Bedrock Guardrails available for prompt-injection detection and content filtering; enable by setting `GUARDRAIL_ID` and `GUARDRAIL_VERSION`
- Outbound HTTP from the tool Lambda restricted to an allow-list of known source-API hostnames, blocking SSRF attempts against internal AWS metadata endpoints
- DynamoDB leads table encrypted with a customer-managed KMS key; TTL configured to expire records after 90 days

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for reporting security issues.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
