import os
import re
import json
from typing import Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from prompt_engineering import (
    MEDICAL_SYSTEM_PROMPT,
    MEDICAL_DISCLAIMER,
    SHORT_DISCLAIMER,
    EMERGENCY_RESPONSE,
    OUT_OF_SCOPE_RESPONSE,
    GREETING_RESPONSE,
    format_context,
)
from conversation_memory import ConversationMemoryManager
from api_client import UniversalLLMClient

load_dotenv()

EMERGENCY_KEYWORDS = [
    'chest pain', 'heart attack', 'can\'t breathe', 'cannot breathe',
    'difficulty breathing', 'severe bleeding', 'unconscious', 'stroke',
    'suicide', 'kill myself', 'want to die', 'overdose', 'poisoning',
    'seizure', 'anaphylaxis', 'choking', 'drowning', 'severe burn',
    'broken bone', 'bone sticking out', 'unresponsive', 'not breathing',
]

ANXIETY_KEYWORDS = [
    'panic', 'panicking', 'scared', 'terrified', 'anxious', 'anxiety attack',
    'heart racing', 'cannot calm down', 'so afraid', 'freaking out', 'overwhelmed',
    'feel like dying', 'doom', 'hyperventilating'
]

GREETING_PATTERNS = [
    r'\b(hi|hello|hey|good morning|good afternoon|good evening|howdy)\b',
    r'\b(how are you|what\'s up|sup|yo)\b',
    r'\b(who are you|what are you|what can you do|help me)\b',
]

OUT_OF_SCOPE_INDICATORS = [
    'weather', 'stock price', 'recipe for cake', 'movie recommendation',
    'sports score', 'programming', 'javascript', 'python code',
    'translate', 'calculator', 'math problem', 'tell me a joke',
    'buy', 'purchase', 'order', 'shopping',
]


class HealthcareChatbot:
    def __init__(self, api_key: Optional[str] = None, provider: str = "auto", model_name: Optional[str] = None):
        self.api_key = (api_key or os.getenv('OPENROUTER_API_KEY', '') or os.getenv('OPENAI_API_KEY', '')).strip()
        self.provider = provider
        self.model_name = (model_name or os.getenv('LLM_MODEL', 'meta-llama/llama-3.3-70b-instruct')).strip()
        self.llm_client = UniversalLLMClient(api_key=self.api_key, provider=self.provider, model=self.model_name)
        self.is_initialized = bool(self.api_key)
        self.initialization_error = None
        self.vector_store = None
        self.embeddings = None
        self.memory = ConversationMemoryManager()
        self.retriever = None
        self._initialize()

    def set_config(self, api_key: str, provider: str = "auto", model_name: Optional[str] = None):
        self.api_key = api_key.strip() if api_key else ""
        self.provider = provider
        if model_name:
            self.model_name = model_name.strip()
        self.llm_client = UniversalLLMClient(api_key=self.api_key, provider=self.provider, model=self.model_name)
        self.is_initialized = bool(self.api_key)
        self._initialize()

    def _initialize(self):
        try:
            if not self.embeddings:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name='all-MiniLM-L6-v2',
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True},
                )

            if not self.vector_store:
                self._initialize_vector_store()

            if self.api_key:
                self.is_initialized = True
                self.initialization_error = None
            else:
                self.is_initialized = False
                self.initialization_error = "API Key not configured."

        except Exception as e:
            self.initialization_error = str(e)
            self.is_initialized = False

    def _initialize_vector_store(self):
        store_path = os.path.join(os.path.dirname(__file__), 'data', 'faiss_medical_store')
        if os.path.exists(store_path):
            self.vector_store = FAISS.load_local(store_path, self.embeddings, allow_dangerous_deserialization=True)
            self.retriever = self.vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
        else:
            self._create_vector_store()

    def _create_vector_store(self):
        kb_path = os.path.join(os.path.dirname(__file__), 'data', 'medical_knowledge_base.json')
        if not os.path.exists(kb_path):
            return

        with open(kb_path, 'r') as f:
            knowledge_base = json.load(f)

        documents = []
        for entry in knowledge_base:
            doc = Document(
                page_content=entry['content'],
                metadata={'id': entry['id'], 'topic': entry['topic'], 'source': entry['source'], 'category': entry['category']},
            )
            documents.append(doc)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=['\n\n', '\n', '. ', ' ', ''])
        chunks = text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

        store_path = os.path.join(os.path.dirname(__file__), 'data', 'faiss_medical_store')
        self.vector_store.save_local(store_path)
        self.retriever = self.vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    def _is_emergency(self, query: str) -> bool:
        return any(keyword in query.lower() for keyword in EMERGENCY_KEYWORDS)

    def _is_anxiety_detected(self, query: str) -> bool:
        return any(keyword in query.lower() for keyword in ANXIETY_KEYWORDS)

    def _is_greeting(self, query: str) -> bool:
        query_lower = query.lower().strip()
        for pattern in GREETING_PATTERNS:
            if re.search(pattern, query_lower) and len(query_lower.split()) <= 6:
                return True
        return False

    def _is_out_of_scope(self, query: str) -> bool:
        return any(indicator in query.lower() for indicator in OUT_OF_SCOPE_INDICATORS)

    def _classify_query(self, query: str) -> str:
        if self._is_emergency(query): return 'emergency'
        if self._is_greeting(query): return 'greeting'
        if self._is_out_of_scope(query): return 'out_of_scope'
        return 'health_query'

    def _retrieve_context(self, query: str) -> Tuple[str, List[Dict]]:
        if not self.retriever: return 'Knowledge base not available.', []
        try:
            docs = self.retriever.get_relevant_documents(query)
            context = format_context(docs)
            sources = [{'topic': d.metadata.get('topic', 'Unknown'), 'source': d.metadata.get('source', 'Unknown'), 'category': d.metadata.get('category', 'general')} for d in docs]
            return context, sources
        except Exception:
            return 'Error retrieving context.', []

    def _generate_health_response(self, query: str, context: str, sources: List[Dict], api_key: str, provider: str, model_name: str, language: str) -> str:
        try:
            system_prompt = MEDICAL_SYSTEM_PROMPT.format(disclaimer=MEDICAL_DISCLAIMER)
            rag_instruction = f'\n## RETRIEVED KNOWLEDGE BASE CONTEXT:\n{context}\n\n## INSTRUCTIONS:\n- Respond fluently in {language}\n- Maintain a warm, highly empathetic, reassuring tone\n- Cite medical sources where applicable\n- Include the medical disclaimer at the end\n'
            full_system = system_prompt + '\n' + rag_instruction

            messages = [{"role": "system", "content": full_system}]
            chat_history = self.memory.get_langchain_history()
            for msg in chat_history[-6:]:
                if hasattr(msg, 'content'):
                    role = "user" if msg.__class__.__name__ == 'HumanMessage' else "assistant"
                    messages.append({"role": role, "content": msg.content})
            messages.append({"role": "user", "content": query})

            response_text = self.llm_client.complete(
                messages=messages,
                api_key=api_key,
                provider=provider,
                model=model_name,
                temperature=0.3,
                max_tokens=2048
            )

            if 'disclaimer' not in response_text.lower():
                response_text += SHORT_DISCLAIMER

            if sources:
                response_text += '\n\n---\n**Sources Referenced:**\n'
                seen = set()
                for src in sources:
                    key = f"{src['topic']} - {src['source']}"
                    if key not in seen:
                        seen.add(key)
                        response_text += f"- _{src['topic']}_ — {src['source']}\n"

            return response_text

        except Exception as e:
            return f'I apologize, but I encountered an issue. Please try rephrasing.\n\n_Technical details: {str(e)}_'

    def chat(self, user_message: str, api_key: Optional[str] = None, provider: str = "auto", model_name: Optional[str] = None, language: str = "English") -> Dict:
        active_key = (api_key or self.api_key or os.getenv('OPENROUTER_API_KEY', '') or os.getenv('OPENAI_API_KEY', '')).strip()
        active_model = (model_name or self.model_name or 'meta-llama/llama-3.3-70b-instruct').strip()
        active_provider = provider or self.provider or "auto"

        if not active_key:
            return {
                'response': '⚠️ **API Key Required:** Please paste your API key in the sidebar and click "Apply Key".',
                'sources': [],
                'query_type': 'error',
                'has_disclaimer': False,
                'anxiety_detected': False
            }

        anxiety_detected = self._is_anxiety_detected(user_message)
        self.memory.add_user_message(user_message)
        query_type = self._classify_query(user_message)

        if query_type == 'emergency': response, sources = EMERGENCY_RESPONSE, []
        elif query_type == 'greeting': response, sources = GREETING_RESPONSE, []
        elif query_type == 'out_of_scope': response, sources = OUT_OF_SCOPE_RESPONSE, []
        else:
            context, sources = self._retrieve_context(user_message)
            response = self._generate_health_response(user_message, context, sources, active_key, active_provider, active_model, language)

        self.memory.add_assistant_message(response, sources)
        self.memory.add_to_langchain_memory(user_message, response)

        return {
            'response': response,
            'sources': sources,
            'query_type': query_type,
            'has_disclaimer': 'disclaimer' in response.lower(),
            'anxiety_detected': anxiety_detected
        }

    def clear_conversation(self):
        self.memory.clear_history()

    def get_conversation_summary(self):
        return self.memory.get_summary()
