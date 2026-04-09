# used it to test wether the api is reachable and which api format message to pass ( not both, top_p and temperature!!)
    python -c "from src.LanguageModels import AnthropicLM; lm = AnthropicLM(model_id='claude-haiku-4-5-20251001'); lm.set_sys_prompt('You are a helpful assistant.'); out = lm.generate(['Say hello in one word.'], temperature=0.0, max_tokens=20); print('OUTPUT:', out)"

More info found here (also how the system prompt overwrite exactly works): docs.claude.com/en/api/messages
    Deprecated. Models released after Claude Opus 4.6 do not support setting temperature. A value of 1.0 of will be accepted for backwards compatibility, all other values will be rejected with a 400 error.