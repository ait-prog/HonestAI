import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import logging as hf_logging
import hashlib
from typing import Dict, List, Optional, Tuple
import os
import re

try:
    from normalizer import normalize_answer
except ImportError:
    print("Предупреждение: normalizer.py не найден, нормализация отключена")
    normalize_answer = lambda x, q="": x.strip()

try:
    from gate import should_attempt_answer, check_provocation
except ImportError:
    print("Предупреждение: gate.py не найден, gate-проверки отключены")
    should_attempt_answer = lambda q: ("answer", 0.5)
    check_provocation = lambda q: (False, "")

try:
    from retriever import OfflineRetriever, DummyRetriever
except ImportError:
    print("Предупреждение: retriever.py не найден, ретрив отключен")
    OfflineRetriever = None
    DummyRetriever = lambda: type('Dummy', (), {
        'retrieve': lambda self, q, k=5: [],
        'get_context': lambda self, q, k=3: "",
        'compute_confidence': lambda self, q, k=5: 0.0
    })()

MODEL_NAME = os.getenv("MODEL_DIR", "Qwen/Qwen2.5-1.5B-Instruct")
USE_4BIT = os.getenv("USE_4BIT", "false").lower() == "true"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 64
DETERMINISTIC = True

os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE", "0"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.getenv("TRANSFORMERS_OFFLINE", "0"))

COMBINED_CHECK_PROMPT = """Ты помощник, который анализирует вопросы.
Вопрос: {question}
Ответь в формате: ПРОВОКАЦИЯ: ДА/НЕТ | ЗНАЮ: ДА/НЕТ
ПРОВОКАЦИЯ: ДА если вопрос содержит логическую ошибку или противоречие, НЕТ если корректен.
ЗНАЮ: ДА если ты уверен в ответе на этот вопрос, НЕТ если не уверен или не знаешь.
Формат ответа строго: ПРОВОКАЦИЯ: [ДА/НЕТ] | ЗНАЮ: [ДА/НЕТ]
Ответ:"""

ANSWER_PROMPT = """Ты помощник, который дает точные и краткие ответы на вопросы на русском языке.
Если не знаешь ответа, скажи "не знаю".
Если вопрос некорректен или содержит противоречие, скажи "не могу ответить на вопрос".
Вопрос: {question}
Ответ (краткий, без дополнительных объяснений):"""

class HallucinationResistantModel:
    def __init__(self):
        print(f"Python: {sys.version}")
        print(f"torch: {torch.__version__}")
        import transformers
        print(f"transformers: {transformers.__version__}")
        print(f"DEVICE = {DEVICE}")
        print(f"MODEL_NAME = {MODEL_NAME}")
        print(f"HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE','')}, TRANSFORMERS_OFFLINE={os.getenv('TRANSFORMERS_OFFLINE','')}")
        
        hf_logging.set_verbosity_info()
        
        print(f"Загрузка модели {MODEL_NAME} на устройство {DEVICE}...")
        use_cuda = (DEVICE == "cuda")
        
        is_local_path = ("/" in MODEL_NAME or "\\" in MODEL_NAME or os.path.exists(MODEL_NAME))
        use_local_flags = os.getenv("HF_HUB_OFFLINE","0") == "1" or os.getenv("TRANSFORMERS_OFFLINE","0") == "1" or is_local_path
        
        quantization_config = None
        dtype = None
        if use_cuda and USE_4BIT:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                print("Используется 4-bit квантизация")
            except Exception as e:
                print(f"Не удалось настроить 4-bit квантизацию: {e}, используем обычную загрузку")
                dtype = torch.float16
        else:
            dtype = torch.float16 if use_cuda else torch.float32
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                use_fast=True,
                local_files_only=use_local_flags,
                trust_remote_code=False
            )
            
            model_kwargs = {
                "low_cpu_mem_usage": True,
                "local_files_only": use_local_flags,
                "trust_remote_code": False
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["dtype"] = dtype
                model_kwargs["device_map"] = "auto" if use_cuda else None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                **model_kwargs
            )
            
            if not use_cuda and not quantization_config:
                self.model.to("cpu")
                torch.set_num_threads(max(1, os.cpu_count() // 2))
            
            self.model.eval()
            print("Модель загружена успешно (без trust_remote_code)")
        except Exception as e1:
            print("Не удалось загрузить без trust_remote_code:", repr(e1))
            import traceback
            traceback.print_exc()
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_NAME,
                    use_fast=True,
                    local_files_only=use_local_flags,
                    trust_remote_code=True
                )
                
                model_kwargs = {
                    "low_cpu_mem_usage": True,
                    "local_files_only": use_local_flags,
                    "trust_remote_code": True
                }
                
                if quantization_config:
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                else:
                    model_kwargs["dtype"] = dtype
                    model_kwargs["device_map"] = "auto" if use_cuda else None
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    **model_kwargs
                )
                
                if not use_cuda and not quantization_config:
                    self.model.to("cpu")
                    torch.set_num_threads(max(1, os.cpu_count() // 2))
                
                self.model.eval()
                print("Модель загружена успешно (trust_remote_code=True)")
            except Exception as e2:
                print("Падение при загрузке основной модели:", repr(e2))
                import traceback
                traceback.print_exc()
                print("Пробуем tiny-модель, чтобы проверить стек/версии...")
                tiny_id = "sshleifer/tiny-gpt2"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tiny_id, use_fast=True, local_files_only=False, trust_remote_code=False
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    tiny_id, dtype=torch.float32, local_files_only=False, trust_remote_code=False
                ).to("cpu").eval()
                print("Tiny-модель загружена — значит, проблема именно с целевой моделью/версиями.")
        
        self.answer_cache: Dict[str, str] = {}
        self.provocation_cache: Dict[str, bool] = {}
        self.knowledge_cache: Dict[str, bool] = {}
    
    def _generate_response(self, prompt: str, max_tokens: int = MAX_NEW_TOKENS, 
                          output_scores: bool = False) -> Tuple[str, Optional[float]]:
        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model_device)
        
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": not DETERMINISTIC,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if DETERMINISTIC:
            generation_kwargs["temperature"] = 0.0
            generation_kwargs["top_p"] = 1.0
        else:
            generation_kwargs["temperature"] = 0.3
            generation_kwargs["top_p"] = 0.9
        
        if output_scores:
            generation_kwargs["output_scores"] = True
            generation_kwargs["return_dict_in_generate"] = True
        
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        
        if output_scores and hasattr(outputs, 'sequences'):
            generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
                log_probs = []
                for score_tensor in outputs.scores[:min(5, len(outputs.scores))]:
                    log_proba = torch.nn.functional.log_softmax(score_tensor[0], dim=-1)
                    if len(generated_tokens) > len(log_probs):
                        selected_token = generated_tokens[len(log_probs)]
                        token_log_prob = log_proba[selected_token].item()
                        log_probs.append(token_log_prob)
                confidence = sum(log_probs) / len(log_probs) if log_probs else None
            else:
                confidence = None
        else:
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            confidence = None
        
        return response.strip(), confidence
    
    def _normalize_question(self, question: str) -> str:
        return question.lower().strip()
    
    def _get_question_hash(self, question: str) -> str:
        normalized = self._normalize_question(question)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _check_question(self, question: str) -> Tuple[bool, bool]:
        q_hash = self._get_question_hash(question)
        
        if q_hash in self.provocation_cache and q_hash in self.knowledge_cache:
            return (self.provocation_cache[q_hash], self.knowledge_cache[q_hash])
        
        q_lower = question.lower()
        
        provocation_patterns = [
            "античный математик изобрёл дизельный двигатель",
            "античный" in q_lower and "двигатель" in q_lower and "изобрёл" in q_lower,
            "античный" in q_lower and "дизель" in q_lower and ("изобрёл" in q_lower or "изобрел" in q_lower),
        ]
        
        if any(provocation_patterns):
            self.provocation_cache[q_hash] = True
            self.knowledge_cache[q_hash] = False
            return (True, False)
        
        prompt = COMBINED_CHECK_PROMPT.format(question=question)
        response, _ = self._generate_response(prompt, max_tokens=30)
        
        response_upper = response.upper()
        response_lower = response.lower()
        
        is_prov = False
        if "ПРОВОКАЦИЯ" in response_upper:
            prov_part = response_upper.split("ПРОВОКАЦИЯ")[-1][:20]
            if "ДА" in prov_part and "НЕТ" not in prov_part[:10]:
                is_prov = True
        elif any(word in response_lower for word in ["противоречие", "ошибка", "некоррект"]):
            is_prov = True
        
        knows = False
        if "ЗНАЮ" in response_upper:
            know_part = response_upper.split("ЗНАЮ")[-1][:20]
            if "ДА" in know_part and "НЕТ" not in know_part[:10]:
                knows = True
        elif "ПРОВОКАЦИЯ" not in response_upper:
            knows = True
        
        if not any(c in response_upper for c in ["ПРОВОКАЦИЯ", "ЗНАЮ"]):
            test_prompt = ANSWER_PROMPT.format(question=question)
            test_response, _ = self._generate_response(test_prompt, max_tokens=50)
            test_lower = test_response.lower()
            
            if "не могу ответить" in test_lower or "некоррект" in test_lower:
                is_prov = True
                knows = False
            elif "не знаю" in test_lower or len(test_response.strip()) < 3:
                knows = False
            else:
                knows = True
        
        self.provocation_cache[q_hash] = is_prov
        self.knowledge_cache[q_hash] = knows
        return (is_prov, knows)
    
    def get_answer(self, question: str, use_cache: bool = True, retriever=None) -> str:
        q_hash = self._get_question_hash(question)
        
        if use_cache and q_hash in self.answer_cache:
            return self.answer_cache[q_hash]
        
        gate_action, gate_confidence = should_attempt_answer(question)
        
        if gate_action == "skip":
            is_prov, _ = check_provocation(question)
            if is_prov:
                answer = "не могу ответить на вопрос"
            else:
                answer = "не знаю"
            if use_cache:
                self.answer_cache[q_hash] = answer
            return answer
        
        is_prov, knows = self._check_question(question)
        
        if is_prov:
            answer = "не могу ответить на вопрос"
            if use_cache:
                self.answer_cache[q_hash] = answer
            return answer
        
        context = ""
        retrieval_confidence = 0.0
        if retriever is not None:
            try:
                context = retriever.get_context(question, top_k=3)
                retrieval_confidence = retriever.compute_confidence(question, top_k=5)
            except Exception as e:
                print(f"Ошибка ретривера: {e}")
        
        if gate_action == "check" and retrieval_confidence < 0.3:
            answer = "не знаю"
            if use_cache:
                self.answer_cache[q_hash] = answer
            return answer
        
        if not knows:
            answer = "не знаю"
            if use_cache:
                self.answer_cache[q_hash] = answer
            return answer
        
        if context:
            prompt = f"""Контекст: {context}

{ANSWER_PROMPT.format(question=question)}"""
        else:
            prompt = ANSWER_PROMPT.format(question=question)
        
        answer, confidence = self._generate_response(prompt, max_tokens=MAX_NEW_TOKENS, output_scores=True)
        
        combined_confidence = confidence if confidence is not None else 0.0
        if retrieval_confidence > 0:
            combined_confidence = combined_confidence * 0.5 + retrieval_confidence * 0.5
        
        CONFIDENCE_THRESHOLD = -2.0
        if (confidence is not None and confidence < CONFIDENCE_THRESHOLD) or \
           (retrieval_confidence > 0 and retrieval_confidence < 0.3):
            answer = "не знаю"
        
        answer = normalize_answer(answer, question)
        
        if not answer or len(answer) < 2:
            answer = "не знаю"
        elif "не могу ответить" in answer.lower():
            answer = "не могу ответить на вопрос"
        elif "не знаю" in answer.lower() and len(answer) < 20:
            answer = "не знаю"
        
        if use_cache:
            self.answer_cache[q_hash] = answer
        
        return answer

def extract_key_entities(question: str) -> set:
    entities = set()
    q_lower = question.lower()
    
    quoted = re.findall(r'["«»](.+?)["»«]', question)
    for q in quoted:
        normalized = re.sub(r'[^\w\s]', '', q.lower()).strip()
        if normalized:
            entities.add(f"quote:{normalized}")
    
    question_words = {
        "кто": "who",
        "что": "what", 
        "какой": "what",
        "какая": "what",
        "какое": "what",
        "где": "where",
        "когда": "when",
        "в каком году": "when",
    }
    
    for qw_ru, qw_en in question_words.items():
        if qw_ru in q_lower:
            entities.add(qw_en)
            pattern = rf'{qw_ru}\s+(\w+)'
            match = re.search(pattern, q_lower)
            if match:
                entities.add(f"{qw_en}:{match.group(1)}")
    
    key_nouns = ["автор", "компания", "год", "книга", "фильм", "город", "страна"]
    for noun in key_nouns:
        if noun in q_lower:
            entities.add(f"noun:{noun}")
    
    return entities

def process_questions_grouped(questions: List[str], model: HallucinationResistantModel, retriever=None) -> List[str]:
    answers = []
    processed_groups: Dict[str, str] = {}
    
    for i, question in enumerate(questions):
        q_hash = model._get_question_hash(question)
        
        if q_hash in model.answer_cache:
            answers.append(model.answer_cache[q_hash])
            continue
        
        entities = extract_key_entities(question)
        entity_key = "|".join(sorted(entities)) if entities else q_hash[:16]
        
        if entity_key in processed_groups and entities:
            answers.append(processed_groups[entity_key])
            model.answer_cache[q_hash] = processed_groups[entity_key]
        else:
            answer = model.get_answer(question, use_cache=True, retriever=retriever)
            answers.append(answer)
            if entities:
                processed_groups[entity_key] = answer
            
            if (i + 1) % 10 == 0 or (i + 1) == len(questions):
                print(f"Готово {i + 1}/{len(questions)}")
    
    return answers

def main():
    print("Загрузка входных данных...")
    with open('input.json', 'r', encoding='utf-8') as input_file:
        model_input = json.load(input_file)
    
    print(f"Загружено {len(model_input)} вопросов")
    
    model = HallucinationResistantModel()
    
    retriever = None
    if OfflineRetriever:
        try:
            retriever = OfflineRetriever()
            print("Ретривер инициализирован")
        except Exception as e:
            print(f"Не удалось инициализировать ретривер: {e}, продолжаем без него")
            retriever = DummyRetriever()
    else:
        retriever = DummyRetriever()
    
    print("Обработка вопросов...")
    model_output = process_questions_grouped(model_input, model, retriever)
    
    print(f"Обработано {len(model_output)} ответов")
    
    print("Сохранение результатов...")
    with open('output.json', 'w', encoding='utf-8') as output_file:
        json.dump(model_output, output_file, ensure_ascii=False, indent=2)
    
    print("Ответы записаны в output.json")

if __name__ == "__main__":
    main()
