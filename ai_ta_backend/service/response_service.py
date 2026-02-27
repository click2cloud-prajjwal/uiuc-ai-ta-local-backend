import os
import logging
from typing import List, Dict, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
import time

load_dotenv()


class ResponseService:
    """Service for generating AI responses using retrieved contexts"""

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
        )

        self.chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
        self.temperature = float(os.getenv("CHAT_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("CHAT_MAX_TOKENS", "1500"))

        # Collection-specific token sizes
        self.collection_max_tokens = {
            "biofarma-collection": 2000,
            "zenith-collection": 4000,
        }

        logging.info("ResponseService initialized")
        logging.info(f"Endpoint: {os.getenv('AZURE_OPENAI_CHAT_ENDPOINT')}")
        logging.info(f"Deployment: {self.chat_deployment}")

    # ------------------------------------------------------------------
    # Main Response
    # ------------------------------------------------------------------
    def generate_response(
        self,
        question: str,
        contexts: List[Dict],
        course_name: str,
        conversation_history: Optional[List[Dict]] = None,
        collection_name: str = "",
    ) -> Dict:

        try:
            # Resolve max_tokens dynamically based on collection
            max_tokens = self.collection_max_tokens.get(collection_name, self.max_tokens)
            logging.info(f"🎯 collection_name='{collection_name}' → max_tokens={max_tokens}")

            translated_question = question

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

            system_prompt = self._build_system_prompt(course_name, collection_name=collection_name)

            messages = [{"role": "system", "content": system_prompt}]

            if conversation_history:
                messages.extend(conversation_history[-5:])

            # Use translated question for RAG
            user_message = self._build_user_message(
                    question,  # always use original question
                    context_text,
                    collection_name=collection_name
                )
            messages.append({"role": "user", "content": user_message})

            logging.info(f"Generating response for: {question[:100]}...")
            logging.info(f"Using deployment: {self.chat_deployment}")

            # Generate
            t_llm = time.monotonic()
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            print(f"⏱️ LLM API call took: {time.monotonic() - t_llm:.2f}s", flush=True)
            print(f"⏱️ Prompt tokens: {response.usage.prompt_tokens}, Completion tokens: {response.usage.completion_tokens}", flush=True)

            answer = response.choices[0].message.content
            logging.info(f"Generated response ({len(answer)} chars)")

            final_answer = answer

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
    def _build_system_prompt(self, course_name: str, collection_name: str = "") -> str:

        if "biofarma" in collection_name.lower():
            return f"""You are a helpful AI assistant for Biofarma.

            You assist users with queries related to pharmaceutical products, financial data, customer records, and any uploaded documents.

            Rules:
            - Provide detailed, thorough responses of 7–8 lines maximum.
            - Be informative but concise — cover all relevant points clearly.
            - Handle sensitive financial or customer data with care — do not speculate beyond what the context provides.
            - No sources or citation tags needed.
            - If the context is insufficient, clearly state what information is missing.
            - Format responses cleanly without excessive newlines.
            - Always respond in the SAME language as the user's question.
            - Do not answer the questions if user asks about celebrity,politics, religion, gibberish texts or any other non-banking topics."""

        elif "zenith" in collection_name.lower():
            return f"""You are a helpful AI assistant for Zenith Bank.

            You assist employees and users with company policies, banking procedures, and internal guidelines.

            Rules:
            - If the user asks about any POLICY, return the FULL and COMPLETE policy text without any trimming, summarizing, or omitting any part. Do not shorten policy content under any circumstances.
            - Never add sources, citations, or [Source] tags.
            - Format responses clearly and professionally.
            - If the context does not contain the requested policy, clearly inform the user.
            - Always respond in the SAME language as the user's question.
            - Whenever the user asks for a holiday calendar, always format the response as a clean, aligned tabular structure (columns with headers and rows), never as multiline or vertically stacked text.
            - Do not answer the questions if user asks about celebrity,politics, religion, gibberish texts or any other non-banking topics.
            """

        else:
            # Default prompt
            return f"""You are a helpful AI teaching assistant for the {course_name} course.

            Rules for every answer:
            - Keep responses short and precise.
            - Maximum 5–6 lines. May be shorter but never longer.
            - No long paragraphs, no long lists, no unnecessary detail.
            - No sources, no citations, no [Source] tags.
            - No newline spam. Keep formatting clean and simple.
            - Always respond in the SAME language as the user's question.
            - If context is missing, say so briefly."""


    # ------------------------------------------------------------------
    def _build_user_message(self, question: str, context: str, collection_name: str = "") -> str:

        if "zenith" in collection_name.lower():
            return f"""Context from company documents:

{context}

---

Employee Question: {question}

Instructions:
- If the user explicitly asks to "show", "give", "return", "list" or "what is the full/complete [policy name]", return the FULL policy text as-is from the context.
- For all other questions, answer concisely in 9-10 lines maximum. Extract only the relevant part.
- Never fabricate information not present in the context.
- If the context doesn't contain the requested information, clearly inform the user."""

        elif "biofarma" in collection_name.lower():
            return f"""Context from Biofarma documents:

{context}

---

User Question: {question}

Please provide a detailed and helpful answer based on the context above in 7-8 lines.
Do not speculate beyond what the context provides.
If the context doesn't contain relevant information, let the user know."""

        else:
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
            for i, ctx in enumerate(contexts[:3], 1):
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