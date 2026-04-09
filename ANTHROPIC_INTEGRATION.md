# Anthropic API Integration

This fork adds support for calling Claude models directly via the **Anthropic API** (in addition to the existing AWS Bedrock path).

## Why

The upstream repo only supports Claude through AWS Bedrock, which requires AWS credentials and session tokens. This change lets you run Claude models with just an `ANTHROPIC_API_KEY`.

## Usage

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

python -m STAC_eval.eval_STAC_benchmark \
    --model_agent claude-sonnet-4-5 \
    --model_planner claude-sonnet-4-5 \
    --model_judge claude-sonnet-4-5 \
    --defense no_defense \
    --batch_size 2
```

> **Note**: a Claude Max subscription does **not** include API credits. Get an API key at https://console.anthropic.com/.

## Model routing rule

Routing is determined by the model ID string:

| Model ID pattern                              | Routes to   |
| --------------------------------------------- | ----------- |
| `claude-sonnet-4-5`, `claude-opus-4-1`, …     | **Anthropic API** (new) |
| `us.anthropic.claude-3-5-sonnet-...-v2:0`     | AWS Bedrock |
| `anthropic.claude-3-sonnet-...-v1:0`          | AWS Bedrock |
| `gpt-4.1`, `o3`, `o4-mini`                    | OpenAI API  |
| anything else                                 | vLLM (GPU)  |

The rule is: **if `model_id.lower().startswith("claude")`, use Anthropic API**. Bedrock IDs contain `anthropic.` before `claude` so they don't match.

## Files changed

| File | Change |
| --- | --- |
| `src/LanguageModels.py` | Added `AnthropicLM` class (mirrors `OpenAILM` but uses the Anthropic SDK). |
| `src/Agents.py`         | Imported `AnthropicLM`, added routing branch, added Anthropic code path in `Agent.step()`. |
| `src/STAC.py`           | Imported `AnthropicLM`, added routing branch in `BaseLM.__init__`. |
| `src/Environments.py`   | Added Anthropic tool config format (`input_schema`) and tool result format (`role: user` + `tool_result` blocks) in both `AgentSafetyBenchEnvironment` and `SHADEArenaEnvironment`. |
| `requirements.txt` / `environment.yml` | Added `anthropic>=0.40.0`. |
| `tests/test_environments.py` | Replaced synthetic `"claude-3-sonnet"` fixture with real Bedrock-format ID `"us.anthropic.claude-3-sonnet-20240229-v1:0"` so the routing rule still sends it to the Bedrock branch. |

## Format differences Anthropic vs Bedrock

The Anthropic API is not just a proxy for Bedrock — the wire format differs:

| Aspect              | Bedrock (converse API)                         | Anthropic API                                |
| ------------------- | ---------------------------------------------- | -------------------------------------------- |
| System prompt       | `system=[{"text": "..."}]`                     | `system="..."` (plain string)                |
| Tool schema key     | `toolSpec.inputSchema.json`                    | `input_schema`                               |
| Tool schema wrapper | `{"tools": [{"toolSpec": {...}}]}`             | `[{...}, {...}]` (plain list)                |
| Assistant tool call | `content: [{"toolUse": {...}}]`                | `content: [{"type": "tool_use", ...}]`       |
| Tool result message | `role: "tool"` + `toolResult` block            | `role: "user"` + `type: "tool_result"` block |

These differences are handled in `src/Environments.py` (tool config + tool result formatting) and `src/Agents.py` (parsing `tool_use` blocks from responses).

## Tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

All 166 unit tests pass (no API keys or GPUs required).
