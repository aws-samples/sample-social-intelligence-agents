# Security

## Reporting Security Issues

If you discover a potential security issue in this project, please notify
AWS/Amazon Security via the
[vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/).
Please do **not** create a public GitHub issue.

## Architecture

```text
Streamlit UI (demo/app.py)
    |
    | HTTPS/TLS 1.2+ (SigV4-signed)
    v
Amazon Bedrock AgentCore Runtime  (agent code, multi-agent orchestration)
    |
    | IAM auth (SigV4)
    v
Amazon Bedrock AgentCore Gateway  (MCP protocol, tool discovery)
    |
    | IAM auth
    v
AWS Lambda  (shared tool handlers)
    |
    | HTTPS/TLS 1.2+
    v
External APIs  (HN, YouTube, dev.to, Reddit, GitHub, etc.)

Amazon DynamoDB  <-- AgentCore Runtime (leads, score rows, frontier claims)
AWS Secrets Manager  <-- Lambda only (CMK-encrypted API keys)
```

## Security Design Considerations

1. **Authentication and authorization** -- IAM authentication with SigV4 signing
   is enforced on Amazon API Gateway, Amazon Bedrock AgentCore Gateway, and
   Amazon Bedrock AgentCore Runtime. No anonymous access is permitted.

2. **Secrets management** -- Four third-party credentials (YouTube Data API,
   Product Hunt, GitHub token, Reddit OAuth2). The CDK stack creates these
   secrets empty and CMK-encrypted under the `social-intel/` prefix. Operators
   populate values post-deploy via `aws secretsmanager put-secret-value`. The `_secrets` helper retrieves values
   at invocation time, does not log the value, fails closed (errors are not cached
   and degrade to the unauthenticated path), and caches for a short, env-tunable
   TTL (`SECRET_CACHE_TTL_SECONDS`, default 600s). Enable automatic rotation via
   `aws secretsmanager rotate-secret` for production.

3. **Least-privilege IAM** -- Each component has scoped IAM policies:
   - Lambda: `secret.grant_read(...)` per secret (exact ARNs, plus `kms:Decrypt`
     on the CMK). The credential-consuming tools run in this Lambda, so it is the
     only role with secret access.
   - AgentCore Runtime: `bedrock:InvokeModel` and
     `bedrock:InvokeModelWithResponseStream` for the configured Claude profiles,
     explicit DynamoDB read/write actions plus `dynamodb:TransactWriteItems` on
     the leads/frontier tables, scoped Memory read/write, and
     inline Bedrock Guardrails through its Converse model calls. Converse still requires
     `bedrock:ApplyGuardrail`, scoped to the exact CDK-created guardrail ARN; it does not
     make a separate `ApplyGuardrail` API call. The Runtime role has no Secrets Manager
     access (it reads no secrets).
   - API Gateway: IAM authorization required on all methods.

4. **Encryption at rest** -- The DynamoDB leads and frontier tables AND the four
   Secrets Manager secrets are encrypted with a single customer-managed KMS key
   (CMK) with key rotation enabled. CDK asset S3 buckets use SSE-S3 by default.

5. **Encryption in transit** -- All AWS SDK calls use HTTPS/TLS 1.2+ by
   default. External API calls via `httpx` use HTTPS with certificate
   validation enabled (`verify=True` is the default). The shared
   `_http.get_with_retry()` helper enforces HTTPS for all outbound requests.

6. **Network security** -- AgentCore Runtime is configured for public network
   access to reach the AgentCore Gateway; the Lambda tool handlers make the
   external API calls. For deployments requiring network isolation, use VPC
   configuration with NAT gateways:
   `network_configuration=agentcore.RuntimeNetworkConfiguration.using_vpc(...)`.

7. **Input validation** -- All tool handlers validate and sanitize inputs
   (length limits, allowed character sets, enum validation). User prompts
   are passed to the LLM but not interpolated into code or queries.

## Threat Model

The table below maps the primary threats for this sample to in-repo controls.
Several controls are configurable through environment variables; set the email review
and grounding options to match your production policy.

| ID | Threat | Severity | Mitigation in this repo |
|---|---|---|---|
| TS001 | Prompt injection to the Bedrock agent | High | `SAFETY_FENCE` untrusted-data framing on all four agents; Bedrock Guardrail (`GUARDRAIL_ID`/`GUARDRAIL_VERSION`) with an InstructionOverride deny-topic. The PROMPT_ATTACK content filter is intentionally not used: it blocked benign multi-step task prompts and the agents' own injection fence; the deny-topic blocks real jailbreaks without that false-positive |
| TS002 | IAM privilege escalation via wildcards | High | Scoped IAM (inference-profile ARN, table ARNs, secret ARNs); scoped AgentCore Memory grant (read/write, no delete); CDK-nag suppressions documented |
| TS003 | SSRF to the AWS metadata endpoint | High | `_http.py` outbound host allow-list rejects any host outside the nine source-API domains, including `169.254.169.254`. Residual: the allow-list matches hostnames, not resolved IPs, so DNS rebinding remains a known risk |
| TS004 | Rotated-secret reuse via cache TTL | High | CMK-encrypted secrets; `grant_read` per-secret (exact ARN + `kms:Decrypt`), no `ListSecrets`; Lambda-only access (Runtime has none); cache TTL 10 min, env-tunable via `SECRET_CACHE_TTL_SECONDS`; fail-closed (errors not cached) |
| TS005 | Unsolicited email without review | Medium | Optional human-in-the-loop (`EMAIL_APPROVAL_REQUIRED=true` stores leads as `pending_review`); compliance footer (`COMPLIANCE_FOOTER_REQUIRED`) |
| TS006 | Data poisoning from external APIs | Medium | Tool output treated as untrusted data; email persistence requires at least two distinct supported sources, `grounding_gate.py` verifies email claims against evidence, and `GROUNDING_MIN_SCORE` blocks low-grounding leads |
| TS007 | Frontier dedup race condition | Medium | `claim_url` uses a DynamoDB conditional write (`attribute_not_exists`); `store_lead` uses one DynamoDB transaction to reserve stable product-name and prospect-ID markers with the lead record |
| TS008 | Guardrail bypass outside the CDK deployment | Medium | The CDK deployment creates a guardrail and injects its ID/version into the Runtime. A standalone deployment must set both variables; add CloudWatch alarms on guardrail filter hits |
| TS009 | DynamoDB scan/query abuse | Medium | Product-name and recency lookups use GSIs, not scans; input limits bound query requests and the runtime role grants only the table APIs required for reads and writes |
| TS010 | API Gateway throttling bypass | Low | Throttling (50 req/s, 100 burst); add AWS WAF on API Gateway for distributed abuse |
| TS011 | Lambda timeout / cost abuse | Low | Lambda 60s timeout; AgentCore lifecycle limits (15 min idle, 8 hr max); node/execution timeouts in the orchestrator |
| TS012 | Credential exposure via CloudWatch | Low | Tool handlers return `upstream_error` (no `str(e)`); skip-list logs counts only; secret values not logged |
| TS013 | Sensitive prompt or evidence exposure in observability data | Medium | Strands and AgentCore telemetry is limited to CloudWatch IAM principals; do not put credentials or sensitive personal data in prompts, evidence, or tool results; application and usage log retention is 90 days |

## Data Classification

| Level | Data Types | Storage | Controls |
|---|---|---|---|
| RESTRICTED | API keys, authentication tokens | AWS Secrets Manager | Customer-managed KMS key, scoped `grant_read` IAM, rotation-ready |
| CONFIDENTIAL | Prospect data, email drafts, scores | Amazon DynamoDB | Customer-managed KMS key, TTL expiry, scoped IAM |
| CONFIDENTIAL | Prompts, source evidence, agent telemetry | Amazon CloudWatch Logs / AgentCore spans | IAM access, Runtime application and usage log retention (90 days), CloudWatch Transaction Search |
| PUBLIC | Tool schemas | Source code repository | Version control, code review |

## Security Guidelines by AWS Service

- **AWS Lambda**: Least-privilege execution role, ARM64 runtime, 60s timeout,
  512 MB memory cap, dedicated CloudWatch log group with one-month retention.
- **Amazon API Gateway**: IAM authentication on all methods, request validation
  enabled, throttling configured, access logging to CloudWatch.
- **Amazon Bedrock AgentCore Gateway**: IAM inbound authorization, MCP protocol
  for tool discovery, Lambda target integration.
- **Amazon Bedrock AgentCore Runtime**: IAM authorization, public network (see
  network security above), lifecycle limits, scoped Bedrock model access,
  `tracing_enabled`, and application/usage log delivery. Strands emits
  OpenTelemetry telemetry natively and AgentCore Runtime exports it; CloudWatch
  Transaction Search is an account-level prerequisite for span search.
- **Amazon DynamoDB**: Customer-managed KMS key encryption, TTL expiry, scoped
  IAM policies (no `dynamodb:*` wildcards), table-level resource ARN restrictions.
- **AWS Secrets Manager**: CDK-created CMK-encrypted secrets, per-secret
  `grant_read` (exact ARN + `kms:Decrypt`), no `ListSecrets` or admin actions,
  Lambda-only access (Runtime role has none).
- **Amazon CloudWatch Logs**: Lambda and API access-log groups use a one-month retention policy;
  Runtime application and usage groups use a 90-day retention policy. Access to
  AgentCore and Strands telemetry must be treated as access to confidential prompt and evidence data.

## Shared Responsibility Model

**AWS responsibilities:**
- Securing Amazon Bedrock AgentCore Runtime infrastructure and model execution
  environment
- Protecting AWS Lambda execution environment and function isolation
- Encrypting data in transit between services (AgentCore Gateway to Lambda)
- Managing IAM service-linked roles and service control policies
- Physical security of data centers and hardware

**Customer responsibilities:**
- Configuring least-privilege IAM policies for all components
- Managing AWS credentials securely (use IAM roles, not long-term keys)
- Rotating API keys in AWS Secrets Manager
- Validating and sanitizing inputs to LLM prompts
- Monitoring CloudWatch Logs and CloudTrail for security events
- Keeping dependencies updated and running security scans
- Reviewing CDK-nag findings before deployment

## Security Scanning

Run these scans before deployment:

```bash
# Python SAST
pip install bandit
bandit -r src/ infra/lambda/ -c pyproject.toml

# Dependency audit: install with the infra extra so CDK dependencies are included
pip install pip-audit
pip-audit

# CDK security checks (runs automatically during cdk synth)
cd infra
cdk synth  # cdk-nag is configured in the CDK app
cd ..

# Semgrep (optional)
semgrep --config auto src/ infra/
```

## Risk Assessment

### Security Risks
- **IAM scope**: CDK L2 constructs for AgentCore may generate broader
  permissions than strictly necessary. Mitigated by CDK-nag suppressions
  with documented justifications.
- **External API dependencies**: Tool handlers call third-party APIs that
  could change or return unexpected data. Mitigated by input validation,
  timeout limits, and error handling.

### Compliance Considerations
- **Data residency**: Amazon Bedrock global inference profiles may route
  requests to multiple AWS Regions. Restrict to a single Region if required.
- **Third-party API terms**: Each external API (Hacker News, YouTube, Reddit,
  etc.) has its own terms of service. Review and comply with rate limits,
  attribution requirements, and data usage restrictions before production use.
- **Email compliance**: Generated outreach emails should include unsubscribe
  mechanisms, physical mailing address, and clear sender identification per
  CAN-SPAM Act requirements before sending via Amazon SES.

## Future Hardening: AgentCore Identity

The third-party credentials currently live in CMK-encrypted Secrets Manager and
are read by the Lambda tool handlers. The managed alternative is **Amazon Bedrock
AgentCore Identity**, which provides a token vault plus `@requires_api_key` and
`@requires_access_token` decorators that inject credentials at runtime without the
values ever entering application logs or LLM context. The Reddit OAuth2
client-credentials flow (hand-rolled in `tools/reddit.py`) is the strongest
candidate, since Identity's M2M OAuth2 provider would replace the manual token
fetch and caching entirely.

This migration is intentionally deferred, not skipped, for one architectural
reason: the Identity decorators resolve credentials from the AgentCore Runtime's
workload-identity context, but the credential-consuming tools run in the **Lambda
behind the Gateway**, which has no workload context. Adopting Identity therefore
requires either moving those tools into the Runtime (changing the documented
Gateway/Lambda topology) or fetching a workload token explicitly in the Lambda.
Until then, the CMK-encrypted, least-privilege, fail-closed Secrets Manager path
is the hardened baseline. To migrate: create credential providers with
`agentcore add credential` (never the MCP create/update tools, which would route
secrets through an LLM), reference them in `agentcore.json`, and decorate the
runtime-side tool functions.
