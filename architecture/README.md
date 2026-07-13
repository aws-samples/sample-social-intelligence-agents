# Architecture diagrams

Editable [draw.io](https://www.drawio.com/) sources and rendered PNGs for the diagrams in the top-level [README](../README.md).

| Diagram | Source | Rendered |
|---|---|---|
| Solution architecture | `architecture-overview.drawio` | `architecture-overview.png` |
| Graph orchestration pattern | `graph-pattern.drawio` | `graph-pattern.png` |
| Swarm orchestration pattern | `swarm-pattern.drawio` | `swarm-pattern.png` |

## Regenerate the PNGs

Open a `.drawio` file in the [draw.io desktop app](https://www.drawio.com/) and export to PNG, or use the CLI:

```bash
drawio --export --format png --scale 2 --border 10 \
  --output architecture-overview.png architecture-overview.drawio
```

Diagrams use the AWS Architecture Icons (`mxgraph.aws4`) shape library, which draw.io bundles by default.
