from sentence_transformers import SentenceTransformer
import torch
import json
import re
from typing import Dict, Any, List, Optional
from .bedrock_client import BedrockLLM
from .utils import get_md5


class ToolRAGModel:
    def __init__(self, rag_model_name, device = "cuda:0"):
        self.rag_model_name = rag_model_name
        self.rag_model = None
        self.tool_desc_embedding = None
        self.tool_name = None
        self.tool_embedding_path = None
        self.device = device 
        self.load_rag_model()

    def load_rag_model(self):
        self.rag_model = SentenceTransformer(self.rag_model_name, device=self.device)
        self.rag_model.max_seq_length = 4096
        self.rag_model.tokenizer.padding_side = "right"

    def load_tool_desc_embedding(self, toolbox):
        self.tool_name, _ = toolbox.refresh_tool_name_desc(
            enable_full_desc=True)
        all_tools_str = [json.dumps(
            each) for each in toolbox.prepare_tool_prompts(toolbox.all_tools)]
        md5_value = get_md5(str(all_tools_str))
        print("get the md value of tools:", md5_value)
        self.tool_embedding_path = self.rag_model_name.split(
            '/')[-1] + "tool_embedding_" + md5_value + ".pt"
        try:
            self.tool_desc_embedding = torch.load(
                self.tool_embedding_path, weights_only=False)
            assert len(self.tool_desc_embedding) == len(
                toolbox.all_tools), "The number of tools in the toolbox is not equal to the number of tool_desc_embedding."
        except:
            self.tool_desc_embedding = None
            print("\033[92mInferring the tool_desc_embedding.\033[0m")
            self.tool_desc_embedding = self.rag_model.encode(
                all_tools_str, prompt="", normalize_embeddings=True
            )
            torch.save(self.tool_desc_embedding, self.tool_embedding_path)
            print("\033[92mFinished inferring the tool_desc_embedding.\033[0m")
            print("\033[91mExiting. Please rerun the code to avoid the OOM issue.\033[0m")
            exit()

    def rag_infer(self, query, top_k=5):
        torch.cuda.empty_cache()
        queries = [query]
        query_embeddings = self.rag_model.encode(
            queries, prompt="", normalize_embeddings=True
        )
        if self.tool_desc_embedding is None:
            print("No tool_desc_embedding")
            exit()
        scores = self.rag_model.similarity(
            query_embeddings, self.tool_desc_embedding)
        top_k = min(top_k, len(self.tool_name))
        top_k_indices = torch.topk(scores, top_k).indices.tolist()[0]
        top_k_tool_names = [self.tool_name[i] for i in top_k_indices]
        return top_k_tool_names


class ToolRAGModelBedrock:
    def __init__(self, rag_model_name, device="cuda:0", bedrock_region=None,
                 pool_size=3, client_kwargs=None):
        self.rag_model_name = rag_model_name
        self.device = device
        self.bedrock_region = bedrock_region
        self.pool_size = pool_size
        self.client_kwargs = client_kwargs or {}
        self._bedrock_llm: Optional[BedrockLLM] = None
        self.tool_prompts: List[Dict[str, Any]] = []
        self.available_tools: List[str] = []

    def _ensure_client(self):
        if self._bedrock_llm is None:
            self._bedrock_llm = BedrockLLM(
                model_id=self.rag_model_name,
                region=self.bedrock_region,
                pool_size=self.pool_size,
                client_kwargs=self.client_kwargs,
            )

    def load_tool_desc_embedding(self, toolbox):
        tool_prompts = toolbox.prepare_tool_prompts(toolbox.all_tools)
        self.tool_prompts = tool_prompts
        self.available_tools = [
            prompt.get("name") or prompt.get("tool_name") or ""
            for prompt in tool_prompts
        ]

    def rag_infer(self, query, top_k=5):
        self._ensure_client()
        if not self.tool_prompts:
            return []

        list_items = []
        for prompt in self.tool_prompts[:200]:
            name = prompt.get("name") or prompt.get("tool_name") or None
            description = prompt.get("description") or prompt.get("prompt") or ""
            arguments = prompt.get("arguments") or {}
            if not name:
                continue
            args_str = ", ".join(f"{k}={v}" for k, v in arguments.items())
            entry = f"- {name}"
            if description:
                entry += f": {description}"
            if args_str:
                entry += f" [{args_str}]"
            list_items.append(entry)
        if not list_items:
            return self.available_tools[:top_k]

        tool_list_text = "\n".join(list_items)
        prompt = f"""
You are a biomedical tool planner with access to the following tools:

{tool_list_text}

Task:
Given the biomedical question below, pick the most relevant {top_k} tools from the list. Circulate your reasoning briefly and return format exactly:
<TOOLS>
Tool: TOOL_NAME_1
Tool: TOOL_NAME_2
...
</TOOLS>

Question:
{query}
"""
        response = self._bedrock_llm.chat(prompt, temperature=0.2, max_tokens=512)
        match = re.search(r"<TOOLS>(.*?)</TOOLS>", response, re.DOTALL | re.IGNORECASE)
        if not match:
            return self.available_tools[:top_k]
        tool_lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
        picked = []
        for line in tool_lines:
            if line.lower().startswith("tool:"):
                name = line.split(":", 1)[1].strip()
                if name and name not in picked:
                    picked.append(name)
                if len(picked) >= top_k:
                    break
        return picked[:top_k] if picked else self.available_tools[:top_k]
