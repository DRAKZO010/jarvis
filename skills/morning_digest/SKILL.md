---
name: morning_digest
version: 1.0.0
description: Generate a comprehensive morning briefing including weather, active tasks, reminders, and system status.
author: Jarvis
tags:
  - digest
  - productivity
  - briefing
steps:
  - tool: get_weather
    args: "{city}"
    output_key: weather
  - tool: get_tasks
    args: ""
    output_key: tasks
  - tool: get_reminders
    args: ""
    output_key: reminders
  - tool: get_system_status
    args: ""
    output_key: system
---

# Morning Digest Skill

When delivering the morning digest:
1. Greet the user warmly according to the time of day.
2. Present current weather conditions and forecast for their location.
3. Summarize urgent pending tasks and scheduled reminders for the day.
4. Report system health (battery level, CPU/memory headroom).
5. Conclude with a helpful motivational statement.

