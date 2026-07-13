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

The architecture follows the Open/Closed Principle; adding a tool requires no changes to existing code:

1. Create `src/social_intelligence/tools/<name>.py` with a `handle(params: dict) -> dict` function:

```python
"""My new API: brief description."""

import httpx

def handle(params: dict) -> dict:
    """Fetch data from the API.

    Args:
        params: query (str), limit (int)
    """
    query = str(params.get("query", ""))[:500]
    limit = max(1, min(int(params.get("limit", 10)), 20))

    resp = httpx.get("https://api.example.com/search", params={"q": query, "limit": limit}, timeout=15.0)
    resp.raise_for_status()

    return {"results": resp.json().get("items", []), "query": query}
```

2. Add one line to `src/social_intelligence/tools/registry.py`:

```python
from . import my_new_tool
ROUTES["my-new-tool"] = my_new_tool.handle
```

3. Add the tool schema to `src/social_intelligence/schemas/tool_schema.json`:

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

4. Update the Lambda handler's `_SCHEMA_TO_ROUTE` map in `infra/lambda/handler.py` if the tool schema name differs from the route key.

The Lambda handler, AgentCore Gateway, and API Gateway pick up the new tool automatically. No other changes are needed.

## Finding Contributions to Work On

Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default
GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'help wanted' issues is a
great place to start.

## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.

## Security Issue Notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Do **not** create a public GitHub issue.

## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.
