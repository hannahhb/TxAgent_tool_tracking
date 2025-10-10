if __name__ == '__main__':
  from txagent import TxAgent
  import os
  os.environ["MKL_THREADING_LAYER"] = "GNU"
  
  
  model_name = 'mims-harvard/TxAgent-T1-Llama-3.1-8B'
  rag_model_name = 'mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B'
  multiagent = False
  max_round = 20
  init_rag_num = 0
  step_rag_num = 10
  
  agent = TxAgent(model_name,
                  rag_model_name,
                  enable_summary=False,
                  device_id = 0)
  agent.init_model()
  
  question = "Given a 50-year-old patient experiencing severe acute pain and considering the use of the newly approved medication, Journavx, how should the dosage be adjusted considering the presence of moderate hepatic impairment?"
  
  response = agent.run_multistep_agent(
      question,
      temperature=0.3,
      max_new_tokens=1024,
      max_token=90240,
      call_agent=multiagent,
      max_round=max_round)
  
  print(f"\033[94m{response}\033[0m")
