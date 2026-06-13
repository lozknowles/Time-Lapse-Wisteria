---

name: stackoverflow-agent-knowledge
description: Use this skill when solving coding, configuration, dependency, runtime, Docker, Linux, API, framework, or debugging problems where a verified Stack Overflow-style answer, known fix, failure signature, or environment-specific solution may help.
license: CC-BY-SA-4.0
compatibility: Works with coding agents that support SKILL.md-style skills. Requires web/API access to Stack Overflow for Agents or a compatible knowledge endpoint.
metadata:
version: "0.2"
owner: "Stack Overflow for Agents"
trust_model: "human-reviewed and agent-verified"
last_updated: "2026-06-13"
---

# Stack Overflow Agent Knowledge Skill

## Purpose

Use this skill to help an agent find, validate, apply, and contribute technical solutions using Stack Overflow-style agent knowledge.

This skill is useful for:

* Debugging errors, logs, stack traces, and failed builds
* Finding known fixes for framework, package, Docker, Linux, CI/CD, API, or runtime problems
* Checking whether a solution applies to the current environment
* Producing safer commands instead of vague prose
* Capturing successful fixes as reusable knowledge

## Before using

First inspect the local project and environment.

Check:

* Operating system and version
* Runtime versions
* Package manager
* Framework versions
* Docker/container state
* Relevant logs
* Exact error messages
* Recently changed files
* Existing project instructions such as `AGENTS.md`, `README.md`, `.env.example`, or local `SKILL.md` files

Never apply a public answer blindly.

## Query strategy

When searching, include:

1. Exact error message
2. Framework/library/tool name
3. Version numbers
4. Operating system
5. Runtime or package manager
6. Relevant command that failed
7. Docker/container context if applicable

Good query shape:

```text
<exact error> <tool/framework> <version> <os> <runtime/package manager>
```

Example:

```text
ForegroundServiceDidNotStartInTimeException Android Capacitor service Pixel 8 Android 15
```

## Evaluate results

Prefer answers with:

* Human approval
* Multiple independent agent verifications
* Recent verification date
* Matching environment
* Reproducible commands
* Clear failure signature
* Rollback instructions
* Known affected versions
* Known fixed versions

Treat answers as lower confidence when:

* They are old and unverified
* They do not mention versions
* They require disabling security checks
* They suggest deleting data
* They involve production credentials
* They conflict with official documentation
* They are written as speculation rather than a tested fix

## Confidence model

When presenting a solution, include:

```yaml
confidence: low | medium | high
reason: "<why this confidence level was chosen>"
matched_environment:
  os: "<detected OS>"
  tool: "<detected tool>"
  version: "<detected version>"
last_verified: "<date if known>"
verified_by: "<humans/agents if known>"
```

## Environment fingerprint

Before recommending a fix, compare the answer against the local environment.

Capture:

```yaml
environment:
  os: ""
  architecture: ""
  shell: ""
  docker: ""
  node: ""
  python: ""
  java: ""
  framework: ""
  package_manager: ""
  gpu: ""
  service_manager: ""
```

If the environment does not match, say so.

## Failure signatures

When an error is found, convert it into a reusable signature.

Example:

```yaml
failure_signature:
  service: "Frigate"
  pattern:
    - "MQTT not authorised"
    - "Unable to connect to broker"
  likely_causes:
    - "MQTT username/password mismatch"
    - "Broker ACL problem"
    - "Frigate using stale config"
  checks:
    - "docker logs frigate --tail 100"
    - "docker logs mosquitto --tail 100"
    - "grep -n \"mqtt:\" config.yml"
  safe_fix:
    - "Verify MQTT credentials"
    - "Restart mosquitto"
    - "Restart Frigate"
```

## Prefer executable fixes

Prefer precise commands over vague instructions.

Bad:

```text
Restart the service and check the logs.
```

Good:

```bash
docker restart frigate
sleep 5
docker logs frigate --tail 100
```

For destructive commands, require explicit confirmation before execution.

Destructive examples:

```bash
rm -rf
docker volume rm
DROP DATABASE
git reset --hard
```

## Response format

When returning an answer to the user, use:

````markdown
### Likely cause
...

### Why this matches your setup
...

### Safe fix
```bash
...
````

### Verify

```bash
...
```

### Rollback

```bash
...
```

### Confidence

High / Medium / Low

````

## Contribution workflow

When a new successful fix is discovered, create a draft knowledge entry.

```yaml
title: ""
problem: ""
failure_signature:
  - ""
environment:
  os: ""
  versions:
    - ""
root_cause: ""
fix:
  commands:
    - ""
verification:
  commands:
    - ""
  observed_result: ""
rollback:
  commands:
    - ""
confidence: ""
source:
  project: ""
  date: ""
status: "draft-human-review-required"
````

Do not publish automatically unless the user explicitly asks.

## Security rules

Never include:

* API keys
* Passwords
* Cookies
* Access tokens
* Private SSH keys
* Private URLs
* Personal data
* `.env` contents
* Internal IP addresses unless the user is working locally and the data stays local

Redact secrets like this:

```text
sk-...REDACTED
password=REDACTED
```

## Local-first preference

Before searching externally, check local project knowledge:

1. `AGENTS.md`
2. Local `SKILL.md`
3. README files
4. Existing scripts
5. Docker compose files
6. Previous logs
7. Local notes
8. Internal knowledge base

Then search Stack Overflow-style agent knowledge.

Then search the wider web or official docs.

## For Lawrence's hpubuntu stack

Prioritise environment-aware fixes for:

* Ubuntu 24.04
* Docker Compose
* Home Assistant
* Frigate
* Mosquitto MQTT
* Jellyfin
* Immich
* NVIDIA Quadro P3000 / CUDA 12.2
* LocalWalks
* Capacitor Android
* Codex CLI
* AnythingLLM
* llama.cpp / GGUF models
* NAS-mounted storage

Assume the user values practical working commands over long theory.
