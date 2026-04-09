"""Unit tests for src/LanguageModels.py — prompt formatting and pure methods.

Tests cover format_prompts, convert_messages_format, and base LM methods
without requiring API keys, network access, or GPUs.

Run:  python -m pytest tests/test_language_models.py -v
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# LM base class
# ---------------------------------------------------------------------------

class TestLMBase:
    def _make_lm(self):
        from src.LanguageModels import LM
        obj = object.__new__(LM)
        obj.model_id = "test-model"
        obj.sys_prompt = ""
        obj.model = None
        return obj

    def test_load_sys_prompt_none(self):
        lm = self._make_lm()
        assert lm.load_sys_prompt(None) == ""

    def test_load_sys_prompt_empty_list(self):
        lm = self._make_lm()
        assert lm.load_sys_prompt([]) == ""

    def test_load_sys_prompt_from_file(self, tmp_path):
        p = tmp_path / "prompt.md"
        p.write_text("You are a helpful assistant.")
        lm = self._make_lm()
        result = lm.load_sys_prompt([str(p)])
        assert "helpful assistant" in result

    def test_load_sys_prompt_multiple_files(self, tmp_path):
        p1 = tmp_path / "a.md"
        p2 = tmp_path / "b.md"
        p1.write_text("Part A.")
        p2.write_text("Part B.")
        lm = self._make_lm()
        result = lm.load_sys_prompt([str(p1), str(p2)])
        assert "Part A" in result
        assert "Part B" in result

    def test_set_sys_prompt(self):
        lm = self._make_lm()
        lm.set_sys_prompt("new prompt")
        assert lm.get_sys_prompt() == "new prompt"

    def test_extend_sys_prompt_string(self):
        lm = self._make_lm()
        lm.sys_prompt = "base"
        lm.extend_sys_prompt(" extended")
        assert lm.sys_prompt == "base extended"

    def test_extend_sys_prompt_list(self):
        lm = self._make_lm()
        lm.sys_prompt = "base"
        lm.extend_sys_prompt([" A", " B"])
        assert lm.sys_prompt == ["base A", "base B"]

    def test_format_output(self):
        lm = self._make_lm()
        result = lm.format_output(42, "hello")
        assert result == {"id": 42, "output": "hello"}

    def test_write_outputs(self, tmp_path):
        lm = self._make_lm()
        out = tmp_path / "out.jsonl"
        lm.write_outputs([{"id": 1, "output": "a"}, {"id": 2, "output": "b"}], str(out))
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == 1


# ---------------------------------------------------------------------------
# OpenAILM.format_prompts
# ---------------------------------------------------------------------------

class TestOpenAILMFormatPrompts:
    def _make_openai_lm(self):
        from src.LanguageModels import OpenAILM
        obj = object.__new__(OpenAILM)
        obj.model_id = "gpt-4.1"
        obj.sys_prompt = "You are a test assistant."
        obj.model = MagicMock()
        return obj

    def test_string_prompt(self):
        lm = self._make_openai_lm()
        result = lm.format_prompts(["Hello"])
        assert len(result) == 1
        assert result[0][0]["role"] == "system"
        assert result[0][0]["content"] == "You are a test assistant."
        assert result[0][1]["role"] == "user"
        assert result[0][1]["content"] == "Hello"

    def test_string_prompt_no_sys(self):
        lm = self._make_openai_lm()
        result = lm.format_prompts(["Hello"], add_sys_prompt=False)
        assert len(result[0]) == 1
        assert result[0][0]["role"] == "user"

    def test_none_prompt(self):
        lm = self._make_openai_lm()
        result = lm.format_prompts([None])
        assert result == [None]

    def test_list_prompt_passthrough(self):
        lm = self._make_openai_lm()
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        result = lm.format_prompts([msgs])
        assert len(result[0]) == 2
        assert result[0][0]["content"] == "hi"

    def test_spotlighting(self):
        lm = self._make_openai_lm()
        result = lm.format_prompts(["hello world"], spotlighting=True)
        user_content = result[0][1]["content"]
        # Spaces should be replaced with \u02c6
        assert " " not in user_content
        assert "\u02c6" in user_content

    def test_spotlighting_on_list_prompt(self):
        lm = self._make_openai_lm()
        msgs = [{"role": "user", "content": "hello world"}, {"role": "assistant", "content": "hi back"}]
        result = lm.format_prompts([msgs], spotlighting=True)
        # Only user messages get spotlighting
        assert "\u02c6" in result[0][0]["content"]
        assert result[0][1]["content"] == "hi back"

    def test_multiple_prompts(self):
        lm = self._make_openai_lm()
        result = lm.format_prompts(["a", "b", "c"])
        assert len(result) == 3
        for i, r in enumerate(result):
            assert r[1]["content"] == ["a", "b", "c"][i]


# ---------------------------------------------------------------------------
# OpenAILM.convert_messages_format
# ---------------------------------------------------------------------------

class TestOpenAILMConvertMessages:
    def _make_openai_lm(self):
        from src.LanguageModels import OpenAILM
        obj = object.__new__(OpenAILM)
        obj.model_id = "gpt-4.1"
        obj.sys_prompt = ""
        obj.model = MagicMock()
        return obj

    def test_assistant_with_tool_calls(self):
        lm = self._make_openai_lm()
        messages = [[
            {"role": "assistant", "content": "Let me check.",
             "tool_calls": [{"id": "tc1", "function": {"name": "search", "arguments": '{"q": "test"}'}}]},
        ]]
        result = lm.convert_messages_format(messages)
        # Should produce two entries: text content + tool call
        assert len(result[0]) == 2
        assert result[0][0]["role"] == "assistant"
        assert result[0][0]["content"] == "Let me check."
        tool_msg = json.loads(result[0][1]["content"])
        assert tool_msg["name"] == "search"

    def test_non_assistant_passthrough(self):
        lm = self._make_openai_lm()
        messages = [[{"role": "user", "content": "hi"}]]
        result = lm.convert_messages_format(messages)
        assert result[0][0] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# BedrockLM prompt formatting
# ---------------------------------------------------------------------------

class TestBedrockLMFormatPrompts:
    def _make_bedrock_lm(self):
        from src.LanguageModels import BedrockLM
        obj = object.__new__(BedrockLM)
        obj.model_id = "us.anthropic.claude-3-sonnet-v1"
        obj.sys_prompt = [{"text": "System prompt."}]
        obj.model = MagicMock()
        obj.region = "us-east-1"
        return obj

    def test_string_prompt(self):
        lm = self._make_bedrock_lm()
        result = lm.format_prompts(["Hello"])
        assert len(result) == 1
        assert result[0][0]["role"] == "user"
        assert result[0][0]["content"] == [{"text": "Hello"}]

    def test_none_prompt(self):
        lm = self._make_bedrock_lm()
        result = lm.format_prompts([None])
        assert result == [None]

    def test_list_prompt_with_tool_calls(self):
        """GPT-format messages with tool_calls should be converted to Bedrock toolUse."""
        lm = self._make_bedrock_lm()
        msgs = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "tc1",
                "function": {"name": "search", "arguments": '{"q": "test"}'}
            }]
        }]
        result = lm.format_prompts([msgs])
        converted = result[0][0]
        assert converted["role"] == "assistant"
        assert "toolUse" in converted["content"][0]
        assert converted["content"][0]["toolUse"]["name"] == "search"

    def test_list_prompt_user_string_content(self):
        lm = self._make_bedrock_lm()
        msgs = [{"role": "user", "content": "test message"}]
        result = lm.format_prompts([msgs])
        assert result[0][0]["content"] == [{"text": "test message"}]


# ---------------------------------------------------------------------------
# BedrockLM.convert_messages_format
# ---------------------------------------------------------------------------

class TestBedrockLMConvertMessages:
    def _make_bedrock_lm(self):
        from src.LanguageModels import BedrockLM
        obj = object.__new__(BedrockLM)
        obj.model_id = "us.anthropic.claude-3-sonnet-v1"
        obj.sys_prompt = [{"text": "System."}]
        obj.model = MagicMock()
        obj.region = "us-east-1"
        return obj

    def test_user_message(self):
        lm = self._make_bedrock_lm()
        messages = [[{"role": "user", "content": "hi"}]]
        result = lm.convert_messages_format(messages)
        assert result[0][0]["content"] == [{"text": "hi"}]

    def test_assistant_with_tool_calls(self):
        lm = self._make_bedrock_lm()
        messages = [[{
            "role": "assistant",
            "content": "Checking...",
            "tool_calls": [{"id": "tc1", "function": {"name": "get_info", "arguments": '{"id": 1}'}}]
        }]]
        result = lm.convert_messages_format(messages)
        assistant_msg = result[0][0]
        assert assistant_msg["role"] == "assistant"
        # Should have text + toolUse in content
        assert any("text" in c for c in assistant_msg["content"])
        assert any("toolUse" in c for c in assistant_msg["content"])

    def test_tool_result_message(self):
        """Tool messages with tool_call_id get converted to user role with text content."""
        lm = self._make_bedrock_lm()
        messages = [[{
            "role": "tool",
            "tool_call_id": "tc1",
            "name": "get_info",
            "content": '{"result": "data"}'
        }]]
        result = lm.convert_messages_format(messages)
        tool_msg = result[0][0]
        assert tool_msg["role"] == "user"
        # The convert_messages_format treats tool messages as user messages with text content
        assert tool_msg["content"] == [{"text": '{"result": "data"}'}]


# ---------------------------------------------------------------------------
# BedrockLM sys prompt handling
# ---------------------------------------------------------------------------

class TestBedrockLMSysPrompt:
    def _make_bedrock_lm(self):
        from src.LanguageModels import BedrockLM
        obj = object.__new__(BedrockLM)
        obj.model_id = "us.anthropic.claude-3-sonnet-v1"
        obj.sys_prompt = [{"text": "Original."}]
        obj.model = MagicMock()
        obj.region = "us-east-1"
        return obj

    def test_set_sys_prompt(self):
        lm = self._make_bedrock_lm()
        lm.set_sys_prompt("New prompt")
        assert lm.sys_prompt == [{"text": "New prompt"}]

    def test_extend_sys_prompt(self):
        lm = self._make_bedrock_lm()
        lm.extend_sys_prompt(" Extra.")
        assert lm.sys_prompt[0]["text"] == "Original. Extra."

    def test_extend_sys_prompt_from_none(self):
        lm = self._make_bedrock_lm()
        lm.sys_prompt = None
        lm.extend_sys_prompt("New prompt")
        assert lm.sys_prompt == [{"text": "New prompt"}]
# ---------------------------------------------------------------------------
# AnthropicLM.format_prompts
# ---------------------------------------------------------------------------
class TestAnthropicLMFormatPrompts:
    def _make_anthropic_lm(self):
          from src.LanguageModels import AnthropicLM
          obj = object.__new__(AnthropicLM)
          obj.model_id = "claude-sonnet-4-5"
          obj.sys_prompt = "You are a test assistant."
          obj.model = MagicMock()
          return obj
    
    def test_string_prompt_no_system_in_messages(self):
          """Anthropic passes system separately, so it must NOT be in the message
  list."""
          lm = self._make_anthropic_lm()
          result = lm.format_prompts(["Hello"])
          assert len(result) == 1
          assert len(result[0]) == 1                       # only one message, not two
          assert result[0][0]["role"] == "user"
          assert result[0][0]["content"] == "Hello"
          # Explicitly: no system role ever appears
          assert all(m["role"] != "system" for m in result[0])
    
    def test_none_prompt(self):
          """None marks an inactive agent and must pass through."""
          lm = self._make_anthropic_lm()
          assert lm.format_prompts([None]) == [None]
    
    def test_list_prompt_passthrough(self):
          """Pre-formatted message lists are deepcopied and returned unchanged."""
          lm = self._make_anthropic_lm()
          msgs = [
              {"role": "user", "content": "hi"},
              {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
          ]
          result = lm.format_prompts([msgs])
          assert len(result[0]) == 2
          assert result[0][0]["content"] == "hi"
          # Deepcopy check: mutating the result doesn't touch the original
          result[0][0]["content"] = "mutated"
          assert msgs[0]["content"] == "hi"
        
    def test_multiple_prompts(self):
          lm = self._make_anthropic_lm()
          result = lm.format_prompts(["a", "b", "c"])
          assert len(result) == 3
          for i, r in enumerate(result):
              assert r[0]["role"] == "user"
              assert r[0]["content"] == ["a", "b", "c"][i]

# ---------------------------------------------------------------------------
# AnthropicLM.convert_messages_format
# ---------------------------------------------------------------------------
class TestAnthropicLMConvertMessages:
    def _make_anthropic_lm(self):
          from src.LanguageModels import AnthropicLM
          obj = object.__new__(AnthropicLM)
          obj.model_id = "claude-sonnet-4-5"
          obj.sys_prompt = ""
          obj.model = MagicMock()
          return obj
    
    def test_system_message_is_dropped(self):
          """Anthropic passes system separately — it must never appear in messages."""
          lm = self._make_anthropic_lm()
          messages = [[
              {"role": "system", "content": "You are X."},
              {"role": "user", "content": "hi"},
          ]]
          result = lm.convert_messages_format(messages)
          assert len(result[0]) == 1
          assert result[0][0]["role"] == "user"

    def test_user_message_passthrough(self):
          lm = self._make_anthropic_lm()
          messages = [[{"role": "user", "content": "hi"}]]
          result = lm.convert_messages_format(messages)
          assert result[0][0]["role"] == "user"
          assert result[0][0]["content"] == "hi"

    def test_assistant_with_tool_calls(self):
          """OpenAI tool_calls → Anthropic tool_use blocks in content list."""
          lm = self._make_anthropic_lm()
          messages = [[{
              "role": "assistant",
              "content": "Let me check.",
              "tool_calls": [{
                  "id": "tc1",
                  "function": {"name": "search", "arguments": '{"q": "test"}'}
              }]
          }]]
          result = lm.convert_messages_format(messages)
          assistant_msg = result[0][0]
          assert assistant_msg["role"] == "assistant"
          # Content is a list of blocks: text block + tool_use block
          blocks = assistant_msg["content"]
          assert any(b.get("type") == "text" for b in blocks)
          tool_use_blocks = [b for b in blocks if b.get("type") == "tool_use"]
          assert len(tool_use_blocks) == 1
          assert tool_use_blocks[0]["name"] == "search"
          assert tool_use_blocks[0]["id"] == "tc1"
          assert tool_use_blocks[0]["input"] == {"q": "test"}   # parsed from JSON string

    def test_assistant_multiple_tool_calls(self):
          """Multiple tool calls in one message → multiple tool_use blocks."""
          lm = self._make_anthropic_lm()
          messages = [[{
              "role": "assistant",
              "tool_calls": [
                  {"id": "tc1", "function": {"name": "a", "arguments": "{}"}},
                  {"id": "tc2", "function": {"name": "b", "arguments": "{}"}},
              ]
          }]]
          result = lm.convert_messages_format(messages)
          tool_use_blocks = [b for b in result[0][0]["content"] if b.get("type") ==
  "tool_use"]
          assert len(tool_use_blocks) == 2
          assert {b["name"] for b in tool_use_blocks} == {"a", "b"}

    def test_tool_role_becomes_user_with_tool_result(self):
          """The most important conversion: role='tool' → role='user' with tool_result
  block."""
          lm = self._make_anthropic_lm()
          messages = [[{
              "role": "tool",
              "tool_call_id": "tc1",
              "name": "search",
              "content": '{"result": "data"}'
          }]]
          result = lm.convert_messages_format(messages)
          msg = result[0][0]
          assert msg["role"] == "user"                         # NOT "tool"
          assert isinstance(msg["content"], list)
          block = msg["content"][0]
          assert block["type"] == "tool_result"
          assert block["tool_use_id"] == "tc1"
          assert block["content"] == '{"result": "data"}'