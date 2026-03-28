if __name__ == '__main__':
    import argparse
    import os
    from txagent import TxAgent
    from txagent.toolrag import ToolRAGModel

    BEDROCK_MODEL_ID = "us.meta.llama3-3-70b-instruct-v1:0"

    parser = argparse.ArgumentParser(description="Run a TxAgent example.")
    parser.add_argument("--bedrock-model",  default=BEDROCK_MODEL_ID,
                        help="Bedrock model ID for the main LLM.")
    parser.add_argument("--rag-model",      default='mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B',
                        help="HuggingFace RAG model.")
    parser.add_argument("--bedrock-region", default=None,
                        help="AWS region (default: AWS_REGION env var or us-east-1).")
    parser.add_argument("--rag-device",     type=int, default=None,
                        help="CUDA device index for the HF RAG model (default: cpu).")
    args = parser.parse_args()

    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["VLLM_USE_V1"] = "0"

    rag_device = f"cuda:{args.rag_device}" if args.rag_device is not None else "cpu"
    region = args.bedrock_region or os.environ.get("AWS_REGION", "us-east-1")

    # Main LLM: AWS Bedrock
    agent = TxAgent(
        model_name=args.bedrock_model,
        rag_model_name=args.rag_model,
        use_bedrock=True,
        bedrock_model_id=args.bedrock_model,
        bedrock_region=region,
        enable_summary=False,
        force_finish=True,
    )

    # RAG: HuggingFace SentenceTransformer (swap before init_model)
    agent.rag_model = ToolRAGModel(args.rag_model, device=rag_device)

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
        max_round=20,
    )

    meta = {
        "txagent": True,
        "tools": getattr(agent, "last_used_tools", None),
        "tool_log_path": getattr(agent, "last_question_log_path", None),
    }

    print(f"\033[94m{response}\033[0m")
    print(meta)
