# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.

## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment

## Contributing via Pull Requests

Contributions via pull requests are much appreciated. Before sending us a pull request, ensure that:

1. You are working against the latest source on the *main* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request:

1. Fork the repository.
2. Modify the source; focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

## How to Add a New Tool

The extension points keep a new tool localized to its module, registry, schema, and
intended agent allow-list:

1. Create `src/social_intelligence/tools/<name>.py` with a `handle(params: dict) -> dict` function:

```python
"""My new API: brief description."""

from ._http import get_with_retry

def handle(params: dict) -> dict:
    """Fetch data from the API.

    Args:
        params: query (str), limit (int)
    """
    query = str(params.get("query", ""))[:500]
    try:
        requested_limit = int(params.get("limit", 10))
    except (TypeError, ValueError):
        requested_limit = 10
    limit = max(1, min(requested_limit, 20))

    resp = get_with_retry("https://api.example.com/search", params={"q": query, "limit": limit})
    resp.raise_for_status()

    return {"results": resp.json().get("items", []), "query": query}
```

2. Add the API hostname to `_ALLOWED_HOSTS` in `src/social_intelligence/tools/_http.py`.
   All external calls must use `get_with_retry()` or `post_with_retry()`; this preserves
   the outbound allow-list, retries, and circuit breaker.

3. Import the module and add a route in `src/social_intelligence/tools/registry.py`:

```python
from . import my_new_tool

ROUTES = {
    # Existing routes...
    "my_new_tool": my_new_tool.handle,
}
```

4. Add the tool schema to `src/social_intelligence/schemas/tool_schema.json`:

```json
{
  "name": "my_new_tool",
  "description": "Brief description of what the tool does.",
  "inputSchema": {
    "type": "object",
    "required": ["query"],
    "properties": {
      "query": {"type": "string", "description": "Search query"},
      "limit": {"type": "integer", "description": "Max results", "default": 10}
    }
  }
}
```

5. Update the Lambda handler's `_SCHEMA_TO_ROUTE` map in `infra/lambda/handler.py` if the tool schema name differs from the route key.

6. Add the schema name to the appropriate allow-list in `entrypoint.py`
   (`_TREND_GATEWAY_TOOL_NAMES` or `_ENRICHMENT_GATEWAY_TOOL_NAMES`) so the
   intended agent can discover it through AgentCore Gateway.

7. Add hermetic unit coverage for validation and error paths, then run
   `python -m pytest tests/ -q --ignore=tests/integration`.

The Lambda handler, AgentCore Gateway, and API Gateway then pick up the new tool
automatically. Keep the agent allow-list narrow: a new tool is not exposed to every
agent by default.

## Finding Contributions to Work On

Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default
GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'help wanted' issues is a
great place to start.

## Community and Security

Read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards. Report vulnerabilities through the process in [SECURITY.md](SECURITY.md), not through a public GitHub issue.

## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.
