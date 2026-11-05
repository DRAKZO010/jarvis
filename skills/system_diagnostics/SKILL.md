---
name: system_diagnostics
version: 1.0.0
description: Collect full system metrics (CPU, RAM, disk, network, uptime, battery) and summarize health.
author: Jarvis
tags:
  - system
  - diagnostics
  - monitoring
steps:
  - tool: get_system_status
    args: ""
    output_key: stats
---

# System Diagnostics Skill

1. Fetch complete hardware and resource metrics using `system` MCP.
2. Evaluate if CPU or Memory usage exceeds 85% threshold.
3. Check disk space headroom across local drives.
4. If on laptop, report battery percentage and charging state.
5. Provide actionable recommendations if bottlenecks are identified.

