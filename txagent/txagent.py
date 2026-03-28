import gradio as gr
import os
import sys
import json
import gc
import math
import re
import logging
import copy
from collections import Counter
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from jinja2 import Template

try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except ImportError:
    LLM = None          # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]
    _VLLM_AVAILABLE = False
import types
from tooluniverse import ToolUniverse
from gradio import ChatMessage
from .bedrock_client import BedrockLLM
from .toolrag import ToolRAGModel, ToolRAGModelBedrock
from .utils import NoRepeatSentenceProcessor, ReasoningTraceChecker, tool_result_format

try:
    import spacy  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    spacy = None

try:
    from spacy.lang.en.stop_words import STOP_WORDS  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    STOP_WORDS = None

try:  # pragma: no cover - optional dependency
    from scispacy.umls_linking import UmlsEntityLinker  # type: ignore
except Exception:
    UmlsEntityLinker = None

try:  # pragma: no cover - optional dependency
    from quickumls import QuickUMLS  # type: ignore
except Exception:
    QuickUMLS = None


logger = logging.getLogger(__name__)


class TxAgent:
    def __init__(self, model_name,
                 rag_model_name,
                 tool_files_dict=None,  # None leads to the default tool files in ToolUniverse
                 enable_finish=True,
                 enable_rag=True,
                 enable_summary=False,
                 init_rag_num=0,
                 step_rag_num=10,
                 summary_mode='step',
                 summary_skip_last_k=0,
                 summary_context_length=None,
                 force_finish=True,
                 avoid_repeat=True,
                 seed=None,
                 enable_checker=False,
                 enable_chat=False,
                 additional_default_tools=None,
                 enable_entity_awareness=False,
                 entity_model_name=None,
                 enable_umls_linking=False,
                 umls_linker_kwargs: Optional[Dict[str, Any]] = None,
                 quickumls_path: Optional[str] = None,
                 umls_max_candidates: int = 3,
                 device_id = None,
                 use_bedrock: bool = False,
                 bedrock_model_id: Optional[str] = None,
                 bedrock_region: Optional[str] = None,
                 bedrock_client_kwargs: Optional[Dict[str, Any]] = None,
                 bedrock_rag_model_id: Optional[str] = None,
                 ):
        self.model_name = model_name
        self.tokenizer = None
        self.terminators = None
        self.rag_model_name = rag_model_name
        self.tool_files_dict = tool_files_dict
        self.model = None
        self.chat_template: Optional[Template] = None
        env_flag = os.environ.get("TXAGENT_USE_BEDROCK", "").strip().lower()
        self.use_bedrock = use_bedrock or env_flag in {"1", "true", "yes"}
        self.bedrock_model_id = bedrock_model_id or os.environ.get(
            "TXAGENT_BEDROCK_MODEL_ID", model_name)
        self.bedrock_region = bedrock_region or os.environ.get("TXAGENT_BEDROCK_REGION")
        self._bedrock_llm: Optional[BedrockLLM] = None
        self._bedrock_client_kwargs = bedrock_client_kwargs or {}
        self.bedrock_rag_model_id = bedrock_rag_model_id or os.environ.get("TXAGENT_BEDROCK_RAG_MODEL_ID")
        if self.bedrock_rag_model_id is None:
            if self.bedrock_model_id:
                self.bedrock_rag_model_id = self.bedrock_model_id
            else:
                self.bedrock_rag_model_id = rag_model_name

        if self.use_bedrock:
            # No local model loaded — device assignments are irrelevant for the main LLM.
            # RAG model (if HF-based) gets device_id if provided, else CPU.
            self.device = "cpu"
            device_rag = f"cuda:{device_id}" if device_id is not None else "cpu"
        else:
            # Both main model and RAG model need GPU.
            # Default to cuda:0; if device_id is explicit, RAG goes on the same device.
            self.device = f"cuda:{device_id}" if device_id is not None else "cuda:0"
            device_rag = self.device

        print(f"[INFO] Main model device: {self.device}")
        print(f"[INFO] RAG model device: {device_rag}")
        
        if self.use_bedrock:
            self.rag_model = ToolRAGModelBedrock(
                self.bedrock_rag_model_id or rag_model_name,
                device=device_rag,
                bedrock_region=self.bedrock_region,
                client_kwargs=self._bedrock_client_kwargs,
            )
        else:
            self.rag_model = ToolRAGModel(rag_model_name, device=device_rag)
        
        

        self.tooluniverse = None
        # self.tool_desc = None
        self.prompt_multi_step = "You are a helpful assistant that will solve problems through detailed, step-by-step reasoning and actions based on your reasoning. Typically, your actions will use the provided functions. You have access to the following functions."
        self.self_prompt = "Strictly follow the instruction."
        self.chat_prompt = "You are helpful assistant to chat with the user."
        self.enable_finish = enable_finish
        self.enable_rag = enable_rag
        self.enable_summary = enable_summary
        self.summary_mode = summary_mode
        self.summary_skip_last_k = summary_skip_last_k
        self.summary_context_length = summary_context_length
        self.init_rag_num = init_rag_num
        self.step_rag_num = step_rag_num
        self.force_finish = force_finish
        self.avoid_repeat = avoid_repeat
        self.seed = seed
        self.enable_checker = enable_checker
        self.additional_default_tools = additional_default_tools
        self.enable_entity_awareness = (enable_entity_awareness or enable_umls_linking) and spacy is not None
        self.enable_umls_linking = enable_umls_linking and self.enable_entity_awareness
        self.entity_model_name = entity_model_name
        self._entity_nlp = None
        self.umls_linker_kwargs = umls_linker_kwargs or {}
        self.quickumls_path = quickumls_path
        self.umls_max_candidates = max(1, umls_max_candidates)
        self._umls_linker = None
        self._quickumls = None
        # base_stopwords = {
           
        # }
        # if STOP_WORDS is not None:
        #     base_stopwords |= {w.lower() for w in STOP_WORDS}
        # self.entity_stopwords = base_stopwords
        default_labels = {
            "CHEMICAL", "DRUG", "DISEASE", "SYMPTOM", "SIGN_OR_SYMPTOM",
            "THERAPEUTIC_PROCEDURE", "DIAGNOSTIC_PROCEDURE", "PATHOLOGICAL_FUNCTION",
            "ANATOMICAL_STRUCTURE", "INJURY_OR_POISONING", "LAB_VALUE", "GENE_OR_GENOME",
        }
        label_overrides = self.umls_linker_kwargs.pop("entity_labels", None)
        if label_overrides:
            default_labels = {lbl.upper() for lbl in label_overrides}
        self.entity_label_whitelist = {lbl.upper() for lbl in default_labels}
        self.last_entity_metadata: List[Dict[str, Any]] = []
        self._tools_used_current: List[Dict[str, Any]] = []
        self._tool_usage_stack: List[List[Dict[str, Any]]] = []
        self.last_used_tools: List[Dict[str, Any]] = []
        self.print_self_values()

    def init_model(self):
        self.load_models()
        self.load_tooluniverse()
        self.load_tool_desc_embedding()

    def print_self_values(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")

    def load_models(self, model_name=None):
        if model_name is not None:
            if model_name == self.model_name:
                return f"The model {model_name} is already loaded."
            self.model_name = model_name

        if self.use_bedrock:
            target_model = self.bedrock_model_id or self.model_name
            self._bedrock_llm = BedrockLLM(
                model_id=target_model,
                region=self.bedrock_region,
                client_kwargs=self._bedrock_client_kwargs,
            )
            self.chat_template = None
            self.tokenizer = None
            self.model = None
            return f"Bedrock model {target_model} initialised successfully."

        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vllm is required for the HuggingFace/local backend. "
                "Install it with: pip install vllm  "
                "Or use the Bedrock backend by passing use_bedrock=True."
            )

        self.model = LLM(model=self.model_name)
        self.chat_template = Template(self.model.get_tokenizer().chat_template)
        self.tokenizer = self.model.get_tokenizer()

        return f"Model {model_name} loaded successfully."

    def load_tooluniverse(self):
        self.tooluniverse = ToolUniverse(tool_files=self.tool_files_dict)
        self.tooluniverse.load_tools()
        special_tools = self.tooluniverse.prepare_tool_prompts(
            self.tooluniverse.tool_category_dicts["special_tools"])
        self.special_tools_name = [tool['name'] for tool in special_tools]

    def load_tool_desc_embedding(self):
        if self.rag_model is None:
            return
        self.rag_model.load_tool_desc_embedding(self.tooluniverse)

    def rag_infer(self, query, top_k=5):
        if self.rag_model is None:
            return []
        return self.rag_model.rag_infer(query, top_k)

    # -------------------- tool usage tracking --------------------
    def _reset_tool_usage(self):
        self._tool_usage_stack.append([])
        self._tools_used_current = self._tool_usage_stack[-1]

    def _record_tool_usage(self, name: str, stage: str, details: Optional[Dict[str, Any]] = None):
        if not self._tool_usage_stack:
            self._reset_tool_usage()
        entry = {
            "name": name,
            "stage": stage,
        }
        if details:
            entry["details"] = details
        self._tool_usage_stack[-1].append(entry)
        self._tools_used_current = self._tool_usage_stack[-1]

    def _finalize_tool_log(self):
        if not self._tool_usage_stack:
            self.last_used_tools = []
            self._tools_used_current = []
            return
        current = self._tool_usage_stack.pop()
        if self._tool_usage_stack:
            self._tools_used_current = self._tool_usage_stack[-1]
            if current:
                self._tool_usage_stack[-1].append({
                    "name": "NestedSession",
                    "stage": "child",
                    "details": {"tools": copy.deepcopy(current)},
                })
        else:
            self.last_used_tools = list(current)
            self._tools_used_current = []

    # -------------------- entity augmentation helpers --------------------
    def _ensure_entity_pipeline(self):
        if not self.enable_entity_awareness:
            return None
        if self._entity_nlp is not None:
            return self._entity_nlp
        model_name = self.entity_model_name or "en_core_sci_sm"
        if spacy is None:
            print("spaCy is unavailable; disabling entity awareness.")
            self.enable_entity_awareness = False
            return None
        try:
            self._entity_nlp = spacy.load(model_name)
            print(f"Loaded spaCy entity model '{model_name}' for query augmentation.")
        except Exception as exc:  # pragma: no cover - runtime path
            print(f"Failed to load spaCy model '{model_name}': {exc}. Disabling entity awareness.")
            self.enable_entity_awareness = False
            self._entity_nlp = None
            return None

        if self.enable_umls_linking:
            self._setup_umls_resources(self._entity_nlp)

        return self._entity_nlp

    def _setup_umls_resources(self, nlp) -> None:
        if not self.enable_umls_linking:
            return

        if UmlsEntityLinker is None and (self.quickumls_path is None or QuickUMLS is None):
            print("UMLS linking requested but dependencies are unavailable; disabling UMLS mode.")
            self.enable_umls_linking = False
            return

        if UmlsEntityLinker is not None:
            try:
                linker_kwargs = dict(self.umls_linker_kwargs)
                pipe_name = linker_kwargs.pop("pipe_name", "scispacy_umls_linker")
                linker_kwargs.setdefault("resolve_abbreviations", True)
                linker_kwargs.setdefault("threshold", 0.85)
                linker_kwargs.setdefault("max_entities_per_mention", self.umls_max_candidates)
                if pipe_name in nlp.pipe_names:
                    linker = nlp.get_pipe(pipe_name)
                else:
                    linker = nlp.add_pipe("scispacy_linker", config=linker_kwargs, name=pipe_name)
                self._umls_linker = linker
                print(f"UMLS linker '{pipe_name}' initialised (SciSpaCy).")
            except Exception as exc:
                print(f"Failed to initialise SciSpaCy UMLS linker: {exc}")
                self._umls_linker = None

        if self.quickumls_path and QuickUMLS is not None and self._quickumls is None:
            try:
                self._quickumls = QuickUMLS(
                    self.quickumls_path,
                    threshold=self.umls_linker_kwargs.get("threshold", 0.85),
                    window=self.umls_linker_kwargs.get("window", 5),
                    min_match_length=self.umls_linker_kwargs.get("min_match_length", 3),
                    best_match=True,
                )
                print("QuickUMLS matcher initialised for entity linking.")
            except Exception as exc:
                print(f"Failed to initialise QuickUMLS at '{self.quickumls_path}': {exc}")
                self._quickumls = None

    def _extract_entities(self, text: str, update_last: bool = True) -> List[Dict[str, Any]]:
        if not text or not self.enable_entity_awareness:
            if update_last:
                self.last_entity_metadata = []
            return []
        nlp = self._ensure_entity_pipeline()
        if nlp is None:
            if update_last:
                self.last_entity_metadata = []
            return []
        doc = nlp(text)
        entities: List[Dict[str, Any]] = []
        for ent in doc.ents:
            label = getattr(ent, "label_", "").upper()
            cleaned_tokens = [
                tok for tok in ent.text.strip().split()
                if tok.lower() not in self.entity_stopwords
            ]
            cleaned = " ".join(cleaned_tokens).strip()
            if not cleaned:
                continue
            linking = self._link_entity(ent)
            synonyms: List[str] = []
            semantic_types: List[str] = []
            for candidate in linking:
                synonyms.extend(candidate.get("aliases", []) or [])
                semantic_types.extend(candidate.get("types", []) or [])
            synonyms = self._deduplicate_preserving_order([s for s in synonyms if s])
            semantic_types = self._deduplicate_preserving_order([t for t in semantic_types if t])
            if self.enable_umls_linking and not linking:
                continue
            if not linking:
                if label and self.entity_label_whitelist and label not in self.entity_label_whitelist:
                    continue
                if cleaned.lower() in self.entity_stopwords:
                    continue
                if len(cleaned) <= 3:
                    continue
            primary = linking[0] if linking else {}
            profile = {
                "surface": cleaned,
                "original": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "primary_cui": primary.get("cui"),
                "canonical_name": primary.get("canonical_name", cleaned),
                "candidates": linking,
                "synonyms": synonyms,
                "semantic_types": semantic_types,
            }
            entities.append(profile)
        if update_last:
            self.last_entity_metadata = entities
        return entities

    def _augment_query_with_entities(self, text: str) -> str:
        if not text:
            return text
        entities = self._extract_entities(text)
        if not entities:
            return text
        unique_profiles: List[Dict[str, Any]] = []
        seen = set()
        for profile in entities:
            key = profile.get("primary_cui") or profile.get("surface", "").lower()
            if key in seen:
                continue
            seen.add(key)
            unique_profiles.append(profile)
        self.last_entity_metadata = unique_profiles
        descriptions: List[str] = []
        expansion_terms: List[str] = []
        for profile in unique_profiles:
            cui = profile.get("primary_cui")
            types = profile.get("semantic_types", [])
            synonyms = profile.get("synonyms", [])
            canonical = profile.get("canonical_name") or profile.get("surface")
            surface = profile.get("surface")
            parts = [canonical]
            if surface and surface.lower() != canonical.lower():
                parts.append(f"mention={surface}")
            if cui:
                parts.append(f"CUI={cui}")
            if types:
                parts.append("types=" + ", ".join(types[:3]))
            if synonyms:
                parts.append("synonyms=" + ", ".join(synonyms[:3]))
            descriptions.append("- " + "; ".join(parts))
            expansion_terms.extend(synonyms[:3])
            if cui:
                expansion_terms.append(cui)
            if canonical and canonical.lower() != (surface or "").lower():
                expansion_terms.append(canonical)
        expansion_terms = self._deduplicate_preserving_order([term for term in expansion_terms if term])
        sections = [text]
        if descriptions:
            sections.append("Entity profiles:\n" + "\n".join(descriptions))
        if expansion_terms:
            sections.append("Related keywords: " + ", ".join(expansion_terms))
        return "\n\n".join(sections)

    def _link_entity(self, ent) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        # SciSpaCy linker results
        if self._umls_linker is not None and hasattr(ent._, "umls_ents"):
            for cui, score in list(getattr(ent._, "umls_ents", []))[: self.umls_max_candidates]:
                entry = {
                    "cui": cui,
                    "score": float(score),
                    "source": "scispacy",
                    "aliases": [],
                    "types": [],
                    "canonical_name": None,
                }
                if cui in self._umls_linker.kb.cui_to_entity:
                    kb_entry = self._umls_linker.kb.cui_to_entity[cui]
                    entry["canonical_name"] = kb_entry.canonical_name
                    entry["aliases"] = list(kb_entry.aliases[:20])
                    entry["types"] = list(kb_entry.types)
                candidates.append(entry)
        # QuickUMLS fallback/augmentation
        if self._quickumls is not None:
            try:
                matches = self._quickumls.match(ent.text, best_match=True)
            except Exception as exc:
                print(f"QuickUMLS linking failed: {exc}")
                matches = []
            for match in matches:
                if not match:
                    continue
                best = match[0]
                entry = {
                    "cui": best.get("cui"),
                    "score": float(best.get("similarity", 0.0)),
                    "source": "quickumls",
                    "canonical_name": best.get("term"),
                    "aliases": list(best.get("aliases", [])) if isinstance(best.get("aliases"), list) else [],
                    "types": list(best.get("semtypes", [])) if isinstance(best.get("semtypes"), (list, set, tuple)) else [],
                }
                candidates.append(entry)
                if len(candidates) >= self.umls_max_candidates:
                    break
        unique: List[Dict[str, Any]] = []
        seen = set()
        for cand in candidates:
            key = (cand.get("cui"), cand.get("source"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(cand)
            if len(unique) >= self.umls_max_candidates:
                break
        return unique

    @staticmethod
    def _deduplicate_preserving_order(items: List[Any]) -> List[Any]:
        seen = set()
        deduped = []
        for item in items:
            if item is None:
                continue
            key = item.lower() if isinstance(item, str) else item
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    # -------------------- uncertainty estimation --------------------
    # def sample_short_answers(self, question: str, k: int = 5,
    #                          temperature: float = 0.7,
    #                          max_new_tokens: int = 256,
    #                          system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    #     """Generate K concise answer samples with rationales for uncertainty analysis."""

    #     if k <= 0:
    #         return []
    #     system_prompt = system_prompt or (
    #         "You are a concise medical reasoning assistant. "
    #         "Answer the question with a short rationale (1-2 sentences) and end with 'Final Answer: <choice or phrase>'."
    #     )
    #     base_conversation = []
    #     base_conversation = self.set_system_prompt(base_conversation, system_prompt)
    #     base_conversation.append({"role": "user", "content": question})

    #     samples = []
    #     for _ in range(k):
    #         output = self.llm_infer(messages=base_conversation,
    #                                 temperature=temperature,
    #                                 tools=None,
    #                                 max_new_tokens=max_new_tokens)
    #         answer, rationale = self._parse_answer_output(output)
    #         samples.append({
    #             "answer": answer,
    #             "rationale": rationale,
    #             "raw": output.strip(),
    #         })
    #     return samples

   
    def _parse_answer_output(self, output: str) -> (str, str):
        text = output.strip()
        match = re.search(r"Final Answer\s*[:\-]?\s*(.+)$", text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            rationale = text[:match.start()].strip()
        else:
            answer = text.split("\n")[-1].strip()
            rationale = text
        return answer, rationale

    def _normalize_answer(self, answer: str) -> str:
        if not answer:
            return ""
        trimmed = answer.strip()
        mc_match = re.search(r"\b([A-E])\b", trimmed.upper())
        if mc_match:
            return mc_match.group(1)
        trimmed = re.sub(r"[^a-z0-9]+", " ", trimmed.lower())
        return trimmed.strip()

    def initialize_tools_prompt(self, call_agent, call_agent_level, message):
        picked_tools_prompt = []
        picked_tools_prompt = self.add_special_tools(
            picked_tools_prompt, call_agent=call_agent)
        if call_agent:
            call_agent_level += 1
            if call_agent_level >= 2:
                call_agent = False

        if not call_agent:
            picked_tools_prompt += self.tool_RAG(
                message=message, rag_num=self.init_rag_num)
        return picked_tools_prompt, call_agent_level

    def initialize_conversation(self, message, conversation=None, history=None):
        if conversation is None:
            conversation = []

        conversation = self.set_system_prompt(
            conversation, self.prompt_multi_step)
        if history is not None:
            if len(history) == 0:
                conversation = []
                print("clear conversation successfully")
            else:
                for i in range(len(history)):
                    if history[i]['role'] == 'user':
                        if i-1 >= 0 and history[i-1]['role'] == 'assistant':
                            conversation.append(
                                {"role": "assistant", "content": history[i-1]['content']})
                        conversation.append(
                            {"role": "user", "content": history[i]['content']})
                    if i == len(history)-1 and history[i]['role'] == 'assistant':
                        conversation.append(
                            {"role": "assistant", "content": history[i]['content']})

        conversation.append({"role": "user", "content": message})

        return conversation

    def tool_RAG(self, message=None,
                 picked_tool_names=None,
                 existing_tools_prompt=[],
                 rag_num=5,
                 return_call_result=False):
        extra_factor = 30  # Factor to retrieve more than rag_num
        if picked_tool_names is None:
            assert picked_tool_names is not None or message is not None
            augmented_message = self._augment_query_with_entities(message)
            picked_tool_names = self.rag_infer(
                augmented_message, top_k=rag_num*extra_factor)
            self._record_tool_usage(
                "Tool_RAG",
                "retrieval",
                {
                    "query": augmented_message,
                    "top_k": rag_num * extra_factor,
                    "entity_metadata": copy.deepcopy(self.last_entity_metadata),
                },
            )

        picked_tool_names_no_special = []
        for tool in picked_tool_names:
            if tool not in self.special_tools_name:
                picked_tool_names_no_special.append(tool)
        picked_tool_names_no_special = picked_tool_names_no_special[:rag_num]
        picked_tool_names = picked_tool_names_no_special[:rag_num]

        picked_tools = self.tooluniverse.get_tool_by_name(picked_tool_names)
        picked_tools_prompt = self.tooluniverse.prepare_tool_prompts(
            picked_tools)
        if picked_tool_names:
            self._record_tool_usage(
                "Tool_RAG",
                "selection",
                {"picked_tools": list(picked_tool_names)},
            )
        if return_call_result:
            return picked_tools_prompt, picked_tool_names
        return picked_tools_prompt

    def add_special_tools(self, tools, call_agent=False):
        if self.enable_finish:
            tools.append(self.tooluniverse.get_one_tool_by_one_name(
                'Finish', return_prompt=True))
            print("Finish tool is added")
        if call_agent:
            tools.append(self.tooluniverse.get_one_tool_by_one_name(
                'CallAgent', return_prompt=True))
            print("CallAgent tool is added")
        else:
            if self.enable_rag:
                tools.append(self.tooluniverse.get_one_tool_by_one_name(
                    'Tool_RAG', return_prompt=True))
                print("Tool_RAG tool is added")

            if self.additional_default_tools is not None:
                for each_tool_name in self.additional_default_tools:
                    tool_prompt = self.tooluniverse.get_one_tool_by_one_name(
                        each_tool_name, return_prompt=True)
                    if tool_prompt is not None:
                        print(f"{each_tool_name} tool is added")
                        tools.append(tool_prompt)
        return tools

    def add_finish_tools(self, tools):
        tools.append(self.tooluniverse.get_one_tool_by_one_name(
            'Finish', return_prompt=True))
        print("Finish tool is added")
        return tools

    def set_system_prompt(self, conversation, sys_prompt):
        if len(conversation) == 0:
            conversation.append(
                {"role": "system", "content": sys_prompt})
        else:
            conversation[0] = {"role": "system", "content": sys_prompt}
        return conversation

    def run_function_call(self, fcall_str,
                          return_message=False,
                          existing_tools_prompt=None,
                          message_for_call_agent=None,
                          call_agent=False,
                          call_agent_level=None,
                          temperature=None):

        function_call_json, message = self.tooluniverse.extract_function_call_json(
            fcall_str, return_message=return_message, verbose=False)
        call_results = []
        special_tool_call = ''
        if function_call_json is not None:
            if isinstance(function_call_json, list):
                for i in range(len(function_call_json)):
                    print("\033[94mTool Call:\033[0m", function_call_json[i])
                    if function_call_json[i]["name"] == 'Finish':
                        special_tool_call = 'Finish'
                        break
                    elif function_call_json[i]["name"] == 'Tool_RAG':
                        new_tools_prompt, call_result = self.tool_RAG(
                            message=message,
                            existing_tools_prompt=existing_tools_prompt,
                            rag_num=self.step_rag_num,
                            return_call_result=True)
                        existing_tools_prompt += new_tools_prompt
                    elif function_call_json[i]["name"] == 'CallAgent':
                        if call_agent_level < 2 and call_agent:
                            solution_plan = function_call_json[i]['arguments']['solution']
                            full_message = (
                                message_for_call_agent +
                                "\nYou must follow the following plan to answer the question: " +
                                str(solution_plan)
                            )
                            call_result = self.run_multistep_agent(
                                full_message, temperature=temperature,
                                max_new_tokens=1024, max_token=99999,
                                call_agent=False, call_agent_level=call_agent_level)
                            call_result = call_result.split(
                                '[FinalAnswer]')[-1].strip()
                        else:
                            call_result = "Error: The CallAgent has been disabled. Please proceed with your reasoning process to solve this question."
                    else:
                        call_result = self.tooluniverse.run_one_function(
                            function_call_json[i])

                    call_id = self.tooluniverse.call_id_gen()
                    function_call_json[i]["call_id"] = call_id
                    self._record_tool_usage(
                        function_call_json[i]["name"],
                        "execution",
                        {
                            "arguments": function_call_json[i].get("arguments", {}),
                            "result_preview": str(call_result)[:500],
                        },
                    )
                    print("\033[94mTool Call Result:\033[0m", call_result)
                    call_results.append({
                        "role": "tool",
                        "content": json.dumps({"content": call_result, "call_id": call_id})
                    })
        else:
            call_results.append({
                "role": "tool",
                "content": json.dumps({"content": "Not a valid function call, please check the function call format."})
            })

        revised_messages = [{
            "role": "assistant",
            "content": message.strip(),
            "tool_calls": json.dumps(function_call_json)
        }] + call_results

        # Yield the final result.
        return revised_messages, existing_tools_prompt, special_tool_call

    def run_function_call_stream(self, fcall_str,
                                 return_message=False,
                                 existing_tools_prompt=None,
                                 message_for_call_agent=None,
                                 call_agent=False,
                                 call_agent_level=None,
                                 temperature=None,
                                 return_gradio_history=True):

        function_call_json, message = self.tooluniverse.extract_function_call_json(
            fcall_str, return_message=return_message, verbose=False)
        call_results = []
        special_tool_call = ''
        if return_gradio_history:
            gradio_history = []
        if function_call_json is not None:
            if isinstance(function_call_json, list):
                for i in range(len(function_call_json)):
                    if function_call_json[i]["name"] == 'Finish':
                        special_tool_call = 'Finish'
                        break
                    elif function_call_json[i]["name"] == 'Tool_RAG':
                        new_tools_prompt, call_result = self.tool_RAG(
                            message=message,
                            existing_tools_prompt=existing_tools_prompt,
                            rag_num=self.step_rag_num,
                            return_call_result=True)
                        existing_tools_prompt += new_tools_prompt
                    elif function_call_json[i]["name"] == 'DirectResponse':
                        call_result = function_call_json[i]['arguments']['respose']
                        special_tool_call = 'DirectResponse'
                    elif function_call_json[i]["name"] == 'RequireClarification':
                        call_result = function_call_json[i]['arguments']['unclear_question']
                        special_tool_call = 'RequireClarification'
                    elif function_call_json[i]["name"] == 'CallAgent':
                        if call_agent_level < 2 and call_agent:
                            solution_plan = function_call_json[i]['arguments']['solution']
                            full_message = (
                                message_for_call_agent +
                                "\nYou must follow the following plan to answer the question: " +
                                str(solution_plan)
                            )
                            sub_agent_task = "Sub TxAgent plan: " + \
                                str(solution_plan)
                            # When streaming, yield responses as they arrive.
                            call_result = yield from self.run_gradio_chat(
                                full_message, history=[], temperature=temperature,
                                max_new_tokens=1024, max_token=99999,
                                call_agent=False, call_agent_level=call_agent_level,
                                conversation=None,
                                sub_agent_task=sub_agent_task)

                            call_result = call_result.split(
                                '[FinalAnswer]')[-1]
                        else:
                            call_result = "Error: The CallAgent has been disabled. Please proceed with your reasoning process to solve this question."
                    else:
                        call_result = self.tooluniverse.run_one_function(
                            function_call_json[i])

                    call_id = self.tooluniverse.call_id_gen()
                    function_call_json[i]["call_id"] = call_id
                    self._record_tool_usage(
                        function_call_json[i]["name"],
                        "execution",
                        {
                            "arguments": function_call_json[i].get("arguments", {}),
                            "result_preview": str(call_result)[:500],
                        },
                    )
                    call_results.append({
                        "role": "tool",
                        "content": json.dumps({"content": call_result, "call_id": call_id})
                    })
                    if return_gradio_history and function_call_json[i]["name"] != 'Finish':
                        if function_call_json[i]["name"] == 'Tool_RAG':
                            gradio_history.append(ChatMessage(role="assistant", content=str(call_result), metadata={
                                                  "title": "🧰 "+function_call_json[i]['name'], "log": str(function_call_json[i]['arguments'])}))

                        else:
                            gradio_history.append(ChatMessage(role="assistant", content=str(call_result), metadata={
                                                  "title": "⚒️ "+function_call_json[i]['name'], "log": str(function_call_json[i]['arguments'])}))
        else:
            call_results.append({
                "role": "tool",
                "content": json.dumps({"content": "Not a valid function call, please check the function call format."})
            })

        revised_messages = [{
            "role": "assistant",
            "content": message.strip(),
            "tool_calls": json.dumps(function_call_json)
        }] + call_results

        # Yield the final result.
        if return_gradio_history:
            return revised_messages, existing_tools_prompt, special_tool_call, gradio_history
        else:
            return revised_messages, existing_tools_prompt, special_tool_call

    def get_answer_based_on_unfinished_reasoning(self, conversation, temperature, max_new_tokens, max_token, outputs=None, return_full_thought=False):
        if conversation[-1]['role'] == 'assisant':
            conversation.append(
                {'role': 'tool', 'content': 'Errors happen during the function call, please come up with the final answer with the current information.'})
        finish_tools_prompt = self.add_finish_tools([])
        final_thought = 'Since I cannot continue reasoning, I will provide the final answer based on the current information and general knowledge.\n\n[FinalAnswer]'
        last_outputs_str = self.llm_infer(messages=conversation,
                                          temperature=temperature,
                                          tools=finish_tools_prompt,
                                          output_begin_string='Since I cannot continue reasoning, I will provide the final answer based on the current information and general knowledge.\n\n[FinalAnswer]',
                                          skip_special_tokens=True,
                                          max_new_tokens=max_new_tokens, max_token=max_token)
        print(last_outputs_str)
        if return_full_thought:
            return final_thought+last_outputs_str
        return last_outputs_str

    def run_multistep_agent(self, message: str,
                            temperature: float,
                            max_new_tokens: int,
                            max_token: int,
                            max_round: int = 20,
                            call_agent=False,
                            call_agent_level=0) -> str:
        """
        Generate a streaming response using the llama3-8b model.
        Args:
            message (str): The input message.
            temperature (float): The temperature for generating the response.
            max_new_tokens (int): The maximum number of new tokens to generate.
        Returns:
            str: The generated response.
        """
        print("\033[1;32;40mstart\033[0m")
        self._reset_tool_usage()

        def _final_return(val):
            self._finalize_tool_log()
            return val

        picked_tools_prompt, call_agent_level = self.initialize_tools_prompt(
            call_agent, call_agent_level, message)
        conversation = self.initialize_conversation(message)

        outputs = []
        last_outputs = []
        next_round = True
        function_call_messages = []
        current_round = 0
        token_overflow = False
        enable_summary = False
        last_status = {}

        if self.enable_checker:
            checker = ReasoningTraceChecker(message, conversation)
        try:
            while next_round and current_round < max_round:
                current_round += 1
                if len(outputs) > 0:
                    function_call_messages, picked_tools_prompt, special_tool_call = self.run_function_call(
                        last_outputs, return_message=True,
                        existing_tools_prompt=picked_tools_prompt,
                        message_for_call_agent=message,
                        call_agent=call_agent,
                        call_agent_level=call_agent_level,
                        temperature=temperature)

                    if special_tool_call == 'Finish':
                        next_round = False
                        conversation.extend(function_call_messages)
                        if isinstance(function_call_messages[0]['content'], types.GeneratorType):
                            function_call_messages[0]['content'] = next(
                                function_call_messages[0]['content'])
                        return _final_return(function_call_messages[0]['content'].split('[FinalAnswer]')[-1])

                    if (self.enable_summary or token_overflow) and not call_agent:
                        if token_overflow:
                            print("token_overflow, using summary")
                        enable_summary = True
                    last_status = self.function_result_summary(
                        conversation, status=last_status, enable_summary=enable_summary)

                    if function_call_messages is not None:
                        conversation.extend(function_call_messages)
                        outputs.append(tool_result_format(
                            function_call_messages))
                    else:
                        next_round = False
                        conversation.extend(
                            [{"role": "assistant", "content": ''.join(last_outputs)}])
                        return _final_return(''.join(last_outputs).replace("</s>", ""))
                if self.enable_checker:
                    good_status, wrong_info = checker.check_conversation()
                    if not good_status:
                        next_round = False
                        print(
                            "Internal error in reasoning: " + wrong_info)
                        break
                last_outputs = []
                outputs.append("### TxAgent:\n")
                last_outputs_str, token_overflow = self.llm_infer(messages=conversation,
                                                                  temperature=temperature,
                                                                  tools=picked_tools_prompt,
                                                                  skip_special_tokens=False,
                                                                  max_new_tokens=max_new_tokens, max_token=max_token,
                                                                  check_token_status=True)
                if last_outputs_str is None:
                    next_round = False
                    print(
                        "The number of tokens exceeds the maximum limit.")
                else:
                    last_outputs.append(last_outputs_str)
            if max_round == current_round:
                print("The number of rounds exceeds the maximum limit!")
            if self.force_finish:
                return _final_return(self.get_answer_based_on_unfinished_reasoning(conversation, temperature, max_new_tokens, max_token))
            else:
                return _final_return(None)

        except Exception as e:
            print(f"Error: {e}")
            if self.force_finish:
                return _final_return(self.get_answer_based_on_unfinished_reasoning(conversation, temperature, max_new_tokens, max_token))
            else:
                return _final_return(None)

    def build_logits_processor(self, messages, llm):
        # Use the tokenizer from the LLM instance.
        tokenizer = llm.get_tokenizer()
        if self.avoid_repeat and len(messages) > 2:
            assistant_messages = []
            for i in range(1, len(messages) + 1):
                if messages[-i]['role'] == 'assistant':
                    assistant_messages.append(messages[-i]['content'])
                    if len(assistant_messages) == 2:
                        break
            forbidden_ids = [tokenizer.encode(
                msg, add_special_tokens=False) for msg in assistant_messages]
            return [NoRepeatSentenceProcessor(forbidden_ids, 5)]
        else:
            return None

    def llm_infer(self, messages, temperature=0.1, tools=None,
                  output_begin_string=None, max_new_tokens=2048,
                  max_token=None, skip_special_tokens=True,
                  model=None, tokenizer=None, terminators=None, seed=None, check_token_status=False):

        if self.use_bedrock:
            if self._bedrock_llm is None:
                self.load_models()
            prompt = self._render_prompt(messages, tools)
            if output_begin_string is not None:
                prompt += output_begin_string
            output = self._bedrock_llm.chat(
                prompt,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            print("\033[92m" + output + "\033[0m")
            if check_token_status and max_token is not None:
                token_overflow = False
                return output, token_overflow
            return output

        if model is None:
            model = self.model

        logits_processor = self.build_logits_processor(messages, model)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            logits_processors=logits_processor,
            seed=seed if seed is not None else self.seed,
        )

        prompt = self._render_prompt(messages, tools)
        if output_begin_string is not None:
            prompt += output_begin_string

        if check_token_status and max_token is not None and self.tokenizer is not None:
            token_overflow = False
            num_input_tokens = len(self.tokenizer.encode(
                prompt, return_tensors="pt")[0])
            if max_token is not None:
                if num_input_tokens > max_token:
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("Number of input tokens before inference:",
                          num_input_tokens)
                    logger.info(
                        "The number of tokens exceeds the maximum limit!!!!")
                    token_overflow = True
                    return None, token_overflow
        output = model.generate(
            prompt,
            sampling_params=sampling_params,
        )
        output = output[0].outputs[0].text
        print("\033[92m" + output + "\033[0m")
        if check_token_status and max_token is not None:
            return output, token_overflow

        return output

    def _render_prompt(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]):
        """
        Render the conversation to a prompt string. Falls back to a simple textual
        template when an LLM chat template is unavailable (e.g., Bedrock mode).
        """
        if self.chat_template is not None:
            return self.chat_template.render(
                messages=messages, tools=tools, add_generation_prompt=True)

        sections: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, list):
                normalized_chunks = []
                for chunk in content:
                    if isinstance(chunk, str):
                        normalized_chunks.append(chunk)
                    else:
                        normalized_chunks.append(json.dumps(chunk, ensure_ascii=False))
                content = "\n".join(normalized_chunks)
            if isinstance(content, types.GeneratorType):
                content = next(content)
            if not isinstance(content, str):
                content = str(content)
            entry = f"{role.upper()}:\n{content.strip()}"
            if message.get("tool_calls"):
                entry += f"\n[TOOL_CALLS]\n{message['tool_calls']}"
            sections.append(entry.strip())

        if tools:
            sections.append("AVAILABLE TOOLS:")
            try:
                sections.append(json.dumps(tools, ensure_ascii=False))
            except TypeError:
                sections.append(str(tools))

        sections.append("ASSISTANT:")
        return "\n\n".join(sections)

    def run_self_agent(self, message: str,
                       temperature: float,
                       max_new_tokens: int,
                       max_token: int) -> str:

        print("\033[1;32;40mstart self agent\033[0m")
        conversation = []
        conversation = self.set_system_prompt(conversation, self.self_prompt)
        conversation.append({"role": "user", "content": message})
        return self.llm_infer(messages=conversation,
                              temperature=temperature,
                              tools=None,
                              max_new_tokens=max_new_tokens, max_token=max_token)

    def run_chat_agent(self, message: str,
                       temperature: float,
                       max_new_tokens: int,
                       max_token: int) -> str:

        print("\033[1;32;40mstart chat agent\033[0m")
        conversation = []
        conversation = self.set_system_prompt(conversation, self.chat_prompt)
        conversation.append({"role": "user", "content": message})
        return self.llm_infer(messages=conversation,
                              temperature=temperature,
                              tools=None,
                              max_new_tokens=max_new_tokens, max_token=max_token)

    def run_format_agent(self, message: str,
                         answer: str,
                         temperature: float,
                         max_new_tokens: int,
                         max_token: int) -> str:

        print("\033[1;32;40mstart format agent\033[0m")
        if '[FinalAnswer]' in answer:
            possible_final_answer = answer.split("[FinalAnswer]")[-1]
        elif "\n\n" in answer:
            possible_final_answer = answer.split("\n\n")[-1]
        else:
            possible_final_answer = answer.strip()
        if len(possible_final_answer) == 1:
            choice = possible_final_answer[0]
            if choice in ['A', 'B', 'C', 'D', 'E']:
                return choice
        elif len(possible_final_answer) > 1:
            if possible_final_answer[1] == ':':
                choice = possible_final_answer[0]
                if choice in ['A', 'B', 'C', 'D', 'E']:
                    print("choice", choice)
                    return choice

        conversation = []
        format_prompt = f"You are helpful assistant to transform the answer of agent to the final answer of 'A', 'B', 'C', 'D'."
        conversation = self.set_system_prompt(conversation, format_prompt)
        conversation.append({"role": "user", "content": message +
                            "\nThe final answer of agent:" + answer + "\n The answer is (must be a letter):"})
        return self.llm_infer(messages=conversation,
                              temperature=temperature,
                              tools=None,
                              max_new_tokens=max_new_tokens, max_token=max_token)

    def run_summary_agent(self, thought_calls: str,
                          function_response: str,
                          temperature: float,
                          max_new_tokens: int,
                          max_token: int) -> str:
        print("\033[1;32;40mSummarized Tool Result:\033[0m")
        generate_tool_result_summary_training_prompt = """Thought and function calls: 
{thought_calls}

Function calls' responses:
\"\"\"
{function_response}
\"\"\"

Based on the Thought and function calls, and the function calls' responses, you need to generate a summary of the function calls' responses that fulfills the requirements of the thought. The summary MUST BE ONE sentence and include all necessary information.

Directly respond with the summarized sentence of the function calls' responses only. 

Generate **one summarized sentence** about "function calls' responses" with necessary information, and respond with a string:
            """.format(thought_calls=thought_calls, function_response=function_response)
        conversation = []
        conversation.append(
            {"role": "user", "content": generate_tool_result_summary_training_prompt})
        output = self.llm_infer(messages=conversation,
                                temperature=temperature,
                                tools=None,
                                max_new_tokens=max_new_tokens, max_token=max_token)

        if '[' in output:
            output = output.split('[')[0]
        return output

    def function_result_summary(self, input_list, status, enable_summary):
        """
        Processes the input list, extracting information from sequences of 'user', 'tool', 'assistant' roles.
        Supports 'length' and 'step' modes, and skips the last 'k' groups.

        Parameters:
            input_list (list): A list of dictionaries containing role and other information.
            summary_skip_last_k (int): Number of groups to skip from the end. Defaults to 0.
            summary_context_length (int): The context length threshold for the 'length' mode.
            last_processed_index (tuple or int): The last processed index.

        Returns:
            list: A list of extracted information from valid sequences.
        """
        if 'tool_call_step' not in status:
            status['tool_call_step'] = 0

        for idx in range(len(input_list)):
            pos_id = len(input_list)-idx-1
            if input_list[pos_id]['role'] == 'assistant':
                if 'tool_calls' in input_list[pos_id]:
                    if 'Tool_RAG' in str(input_list[pos_id]['tool_calls']):
                        status['tool_call_step'] += 1
                break

        if 'step' in status:
            status['step'] += 1
        else:
            status['step'] = 0

        if not enable_summary:
            return status

        if 'summarized_index' not in status:
            status['summarized_index'] = 0

        if 'summarized_step' not in status:
            status['summarized_step'] = 0

        if 'previous_length' not in status:
            status['previous_length'] = 0

        if 'history' not in status:
            status['history'] = []

        function_response = ''
        idx = 0
        current_summarized_index = status['summarized_index']

        status['history'].append(self.summary_mode == 'step' and status['summarized_step']
                                 < status['step']-status['tool_call_step']-self.summary_skip_last_k)

        idx = current_summarized_index
        while idx < len(input_list):
            if (self.summary_mode == 'step' and status['summarized_step'] < status['step']-status['tool_call_step']-self.summary_skip_last_k) or (self.summary_mode == 'length' and status['previous_length'] > self.summary_context_length):

                if input_list[idx]['role'] == 'assistant':
                    if 'Tool_RAG' in str(input_list[idx]['tool_calls']):
                        this_thought_calls = None
                    else:
                        if len(function_response) != 0:
                            print("internal summary")
                            status['summarized_step'] += 1
                            result_summary = self.run_summary_agent(
                                thought_calls=this_thought_calls,
                                function_response=function_response,
                                temperature=0.1,
                                max_new_tokens=1024,
                                max_token=99999
                            )

                            input_list.insert(
                                last_call_idx+1, {'role': 'tool', 'content': result_summary})
                            status['summarized_index'] = last_call_idx + 2
                            idx += 1

                        last_call_idx = idx
                        this_thought_calls = input_list[idx]['content'] + \
                            input_list[idx]['tool_calls']
                        function_response = ''

                elif input_list[idx]['role'] == 'tool' and this_thought_calls is not None:
                    function_response += input_list[idx]['content']
                    del input_list[idx]
                    idx -= 1

            else:
                break
            idx += 1

        if len(function_response) != 0:
            status['summarized_step'] += 1
            result_summary = self.run_summary_agent(
                thought_calls=this_thought_calls,
                function_response=function_response,
                temperature=0.1,
                max_new_tokens=1024,
                max_token=99999
            )

            tool_calls = json.loads(input_list[last_call_idx]['tool_calls'])
            for tool_call in tool_calls:
                del tool_call['call_id']
            input_list[last_call_idx]['tool_calls'] = json.dumps(tool_calls)
            input_list.insert(
                last_call_idx+1, {'role': 'tool', 'content': result_summary})
            status['summarized_index'] = last_call_idx + 2

        return status

    # Following are Gradio related functions

    # General update method that accepts any new arguments through kwargs
    def update_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Return the updated attributes
        updated_attributes = {key: value for key,
                              value in kwargs.items() if hasattr(self, key)}
        return updated_attributes

    def run_gradio_chat(self, message: str,
                        history: list,
                        temperature: float,
                        max_new_tokens: int,
                        max_token: int,
                        call_agent: bool,
                        conversation: gr.State,
                        max_round: int = 20,
                        seed: int = None,
                        call_agent_level: int = 0,
                        sub_agent_task: str = None) -> str:
        """
        Generate a streaming response using the llama3-8b model.
        Args:
            message (str): The input message.
            history (list): The conversation history used by ChatInterface.
            temperature (float): The temperature for generating the response.
            max_new_tokens (int): The maximum number of new tokens to generate.
        Returns:
            str: The generated response.
        """
        print("\033[1;32;40mstart\033[0m")
        print("len(message)", len(message))
        if len(message) <= 10:
            yield "Hi, I am TxAgent, an assistant for answering biomedical questions. Please provide a valid message with a string longer than 10 characters."
            return "Please provide a valid message."
        outputs = []
        outputs_str = ''
        last_outputs = []

        picked_tools_prompt, call_agent_level = self.initialize_tools_prompt(
            call_agent,
            call_agent_level,
            message)

        conversation = self.initialize_conversation(
            message,
            conversation=conversation,
            history=history)
        history = []

        next_round = True
        function_call_messages = []
        current_round = 0
        enable_summary = False
        last_status = {}  # for summary
        token_overflow = False
        if self.enable_checker:
            checker = ReasoningTraceChecker(
                message, conversation, init_index=len(conversation))

        try:
            while next_round and current_round < max_round:
                current_round += 1
                if len(last_outputs) > 0:
                    function_call_messages, picked_tools_prompt, special_tool_call, current_gradio_history = yield from self.run_function_call_stream(
                        last_outputs, return_message=True,
                        existing_tools_prompt=picked_tools_prompt,
                        message_for_call_agent=message,
                        call_agent=call_agent,
                        call_agent_level=call_agent_level,
                        temperature=temperature)
                    history.extend(current_gradio_history)
                    if special_tool_call == 'Finish':
                        yield history
                        next_round = False
                        conversation.extend(function_call_messages)
                        return function_call_messages[0]['content']
                    elif special_tool_call == 'RequireClarification' or special_tool_call == 'DirectResponse':
                        history.append(
                            ChatMessage(role="assistant", content=history[-1].content))
                        yield history
                        next_round = False
                        return history[-1].content
                    if (self.enable_summary or token_overflow) and not call_agent:
                        if token_overflow:
                            print("token_overflow, using summary")
                        enable_summary = True
                    last_status = self.function_result_summary(
                        conversation, status=last_status,
                        enable_summary=enable_summary)
                    if function_call_messages is not None:
                        conversation.extend(function_call_messages)
                        formated_md_function_call_messages = tool_result_format(
                            function_call_messages)
                        yield history
                    else:
                        next_round = False
                        conversation.extend(
                            [{"role": "assistant", "content": ''.join(last_outputs)}])
                        return ''.join(last_outputs).replace("</s>", "")
                if self.enable_checker:
                    good_status, wrong_info = checker.check_conversation()
                    if not good_status:
                        next_round = False
                        print("Internal error in reasoning: " + wrong_info)
                        break
                last_outputs = []
                last_outputs_str, token_overflow = self.llm_infer(
                    messages=conversation,
                    temperature=temperature,
                    tools=picked_tools_prompt,
                    skip_special_tokens=False,
                    max_new_tokens=max_new_tokens,
                    max_token=max_token,
                    seed=seed,
                    check_token_status=True)
                last_thought = last_outputs_str.split("[TOOL_CALLS]")[0]
                for each in history:
                    if each.metadata is not None:
                        each.metadata['status'] = 'done'
                if '[FinalAnswer]' in last_thought:
                    final_thought, final_answer = last_thought.split(
                        '[FinalAnswer]')
                    history.append(
                        ChatMessage(role="assistant",
                                    content=final_thought.strip())
                    )
                    yield history
                    history.append(
                        ChatMessage(
                            role="assistant", content="**Answer**:\n"+final_answer.strip())
                    )
                    yield history
                else:
                    history.append(ChatMessage(
                        role="assistant", content=last_thought))
                    yield history

                last_outputs.append(last_outputs_str)

            if self.force_finish:
                last_outputs_str = self.get_answer_based_on_unfinished_reasoning(
                    conversation, temperature, max_new_tokens, max_token, return_full_thought=True)
                for each in history:
                    if each.metadata is not None:
                        each.metadata['status'] = 'done'

                final_thought, final_answer = last_outputs_str.split('[FinalAnswer]')
                history.append(
                    ChatMessage(role="assistant",
                                content=final_thought.strip())
                )
                yield history
                history.append(
                    ChatMessage(
                        role="assistant", content="**Answer**:\n"+final_answer.strip())
                )
                yield history
            else:
                yield "The number of rounds exceeds the maximum limit!"

        except Exception as e:
            print(f"Error: {e}")
            if self.force_finish:
                last_outputs_str = self.get_answer_based_on_unfinished_reasoning(
                    conversation,
                    temperature,
                    max_new_tokens,
                    max_token,
                    return_full_thought=True)
                for each in history:
                    if each.metadata is not None:
                        each.metadata['status'] = 'done'

                final_thought, final_answer = last_outputs_str.split(
                    '[FinalAnswer]')
                history.append(
                    ChatMessage(role="assistant",
                                content=final_thought.strip())
                )
                yield history
                history.append(
                    ChatMessage(
                        role="assistant", content="**Answer**:\n"+final_answer.strip())
                )
                yield history
            else:
                return None
