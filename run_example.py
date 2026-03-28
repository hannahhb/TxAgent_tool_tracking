if __name__ == '__main__':
  import argparse
  from txagent import TxAgent
  import os

  parser = argparse.ArgumentParser(description="Run a TxAgent example.")
  parser.add_argument("--model", default='mims-harvard/TxAgent-T1-Llama-3.1-8B')
  parser.add_argument("--rag-model", default='mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B')
  parser.add_argument("--use-bedrock", action="store_true",
                      help="Route both chat and RAG through Bedrock.")
  parser.add_argument("--bedrock-model", default=None,
                      help="Bedrock model ID to use (overrides model name).")
  parser.add_argument("--bedrock-region", default=None,
                      help="Override AWS region for Bedrock.")
  parser.add_argument("--device-id", type=int, default=1)
  args = parser.parse_args()

  os.environ["MKL_THREADING_LAYER"] = "GNU"
  os.environ["VLLM_USE_V1"] = "0"

  agent = TxAgent(
      args.model,
      args.rag_model,
      enable_summary=False,
      device_id=args.device_id,
      use_bedrock=args.use_bedrock,
      bedrock_model_id=args.bedrock_model,
      bedrock_region=args.bedrock_region,
  )
  agent.init_model()

  question = (
      "Given a 50-year-old patient experiencing severe acute pain and "
      "considering the use of the newly approved medication, Journavx, "
      "how should the dosage be adjusted considering the presence of "
      "moderate hepatic impairment?"
  )

  response = agent.run_multistep_agent(
      question,
      temperature=0.3,
      max_new_tokens=1024,
      max_token=90240,
      call_agent=False,
      max_round=20)

  tools_final = getattr(agent, "last_used_tools", None)
  tool_log_path = getattr(agent, "last_question_log_path", None)
  meta = {
      "txagent": True,
      "tools": tools_final,               # structured list (finalized tree)
      "tool_log_path": tool_log_path,     # JSON file if you implemented persistence
  }

  print(f"\033[94m{response}\033[0m")
  print(meta)
