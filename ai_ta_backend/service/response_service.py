import os
import logging
from typing import List, Dict, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


class ResponseService:
    """Service for generating AI responses using retrieved contexts"""

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
        )

        self.chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
        self.temperature = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("CHAT_MAX_TOKENS", "1500"))

        logging.info("ResponseService initialized")
        logging.info(f"Endpoint: {os.getenv('AZURE_OPENAI_CHAT_ENDPOINT')}")
        logging.info(f"Deployment: {self.chat_deployment}")


    # ------------------------------------------------------------------
    # Language Detection
    # ------------------------------------------------------------------
    def detect_language(self, text: str) -> str:
        """Detect ISO language code: hi/mr/pa/en"""
        try:
            resp = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": "Detect the language of the text. Answer only ISO code: hi, mr, pa, en."},
                    {"role": "user", "content": text}
                ],
                temperature=0
            )
            return resp.choices[0].message.content.strip().lower()
        except Exception as e:
            logging.error(f"Language detection failed: {e}")
            return "en"


    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    def translate(self, text: str, target_lang: str) -> str:
        """Translate text to target language"""
        try:
            resp = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": f"Translate the following text into {target_lang}. Respond only with the translated text."
                    },
                    {"role": "user", "content": text}
                ],
                temperature=0
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Translation failed: {e}")
            return text


    # ------------------------------------------------------------------
    # Main Response
    # ------------------------------------------------------------------
    def generate_response(
        self,
        question: str,
        contexts: List[Dict],
        course_name: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:

        try:
            # 1. Detect user query language
            original_query_lang = self.detect_language(question)
            logging.info(f"Query language detected: {original_query_lang}")

            # 2. Determine document language from retrieved contexts
            doc_lang = original_query_lang
            if len(contexts) > 0:
                first_group = contexts[0].get("doc_groups")
                if isinstance(first_group, list) and len(first_group) > 0:
                    doc_lang = first_group[0]
                    logging.info(f"Document language detected: {doc_lang}")

            # 3. Translate query to document language if needed
            translated_question = question
            if doc_lang != original_query_lang:
                logging.info(f"Translating query {original_query_lang} -> {doc_lang}")
                translated_question = self.translate(question, doc_lang)

            # Build context string from retrieved chunks
            context_parts = []
            for i, ctx in enumerate(contexts[:5], 1):
                if isinstance(ctx, dict):
                    text = ctx.get('text', ctx.get('page_content', ''))
                    source = ctx.get('readable_filename', 'Unknown')
                    page = ctx.get('pagenumber', '')
                else:
                    text = str(ctx)
                    source = "Unknown"
                    page = ""

                source_info = f"{source}"
                if page:
                    source_info += f" (page {page})"

                context_parts.append(f"[Source {i} - {source_info}]:\n{text}")

            context_text = "\n\n".join(context_parts)

            system_prompt = self._build_system_prompt(course_name)

            messages = [{"role": "system", "content": system_prompt}]

            if conversation_history:
                messages.extend(conversation_history[-5:])

            # Use translated question for RAG
            user_message = self._build_user_message(translated_question, context_text)
            messages.append({"role": "user", "content": user_message})

            logging.info(f"Generating response for: {translated_question[:100]}...")
            logging.info(f"Using deployment: {self.chat_deployment}")

            # Generate
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer = response.choices[0].message.content
            logging.info(f"Generated response ({len(answer)} chars)")

            # 4. Detect answer language
            answer_lang = self.detect_language(answer)

            # 5. Translate answer back to original query language
            final_answer = answer
            if answer_lang != original_query_lang:
                logging.info(f"Translating answer {answer_lang} -> {original_query_lang}")
                final_answer = self.translate(answer, original_query_lang)

            result = {
                "answer": final_answer,
                "sources_used": len(contexts),
                "model": self.chat_deployment,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

            return result

        except Exception as e:
            logging.error(f"Error generating response: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise


    # ------------------------------------------------------------------
    def _build_system_prompt(self, course_name: str) -> str:
        return f"""You are a helpful AI teaching assistant for the {course_name} course.

Rules for every answer:
- Keep responses short and precise.
- Maximum 5–6 lines. May be shorter but never longer.
- No long paragraphs, no long lists, no unnecessary detail.
- No sources, no citations, no [Source] tags.
- No newline spam. Keep formatting clean and simple.
- If context is missing, say so briefly."""


    # ------------------------------------------------------------------
    def _build_user_message(self, question: str, context: str) -> str:
        return f"""Context from course materials:

{context}

---

Student Question: {question}

Please provide a helpful answer based on the context above. 
Please answer in a short, precise 5–6 line maximum format, without sources or citations.
If the context doesn't contain relevant information, let the student know."""


    def generate_streaming_response(
        self,
        question: str,
        contexts: List[Dict],
        course_name: str
    ):
        """
        Generate streaming response for real-time display
        Yields chunks of text as they're generated
        """
        try:
            # Build context
            context_parts = []
            for i, ctx in enumerate(contexts[:5], 1):
                text = ctx.get('text', ctx.get('page_content', ''))
                source = ctx.get('readable_filename', 'Unknown')
                context_parts.append(f"[Source {i} - {source}]:\n{text}")

            context_text = "\n\n".join(context_parts)

            # Build messages
            messages = [
                {"role": "system", "content": self._build_system_prompt(course_name)},
                {"role": "user", "content": self._build_user_message(question, context_text)}
            ]

            # Stream response
            stream = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    yield token

        except Exception as e:
            logging.error(f"❌ Error in streaming response: {e}")
            yield f"\n\n[Error: {str(e)}]"