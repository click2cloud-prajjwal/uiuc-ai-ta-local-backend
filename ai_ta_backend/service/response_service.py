import os
import logging
from typing import List, Dict, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv()


class ResponseService:
    """Service for generating AI responses using retrieved contexts"""

    def __init__(self):
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
        )

        # Use your deployment name
        self.chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
        self.temperature = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("CHAT_MAX_TOKENS", "1500"))


        logging.info(f"✅ ResponseService initialized")
        logging.info(f"   Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
        logging.info(f"   Deployment: {self.chat_deployment}")
        logging.info(f"   API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")


    def generate_response(
        self,
        question: str,
        contexts: List[Dict],
        course_name: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """Generate AI response using retrieved contexts"""
        try:
            # Build context string from retrieved chunks
            context_parts = []
            for i, ctx in enumerate(contexts[:5], 1):
                if isinstance(ctx, dict):
                    text = ctx.get('text', ctx.get('page_content', ''))
                    source = ctx.get('readable_filename', 'Unknown')
                    page = ctx.get('pagenumber', '')
                else:
                    # Handle case where ctx is a plain string
                    text = str(ctx)
                    source = "Unknown"
                    page = ""

                source_info = f"{source}"
                if page:
                    source_info += f" (page {page})"

                context_parts.append(f"[Source {i} - {source_info}]:\n{text}")

            context_text = "\n\n".join(context_parts)
            # Build system prompt
            system_prompt = self._build_system_prompt(course_name)

            # Build conversation messages
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history[-5:])

            # Add current question with context
            user_message = self._build_user_message(question, context_text)
            messages.append({"role": "user", "content": user_message})

            logging.info(f"🤖 Generating response for: {question[:100]}...")
            logging.info(f"   Using deployment: {self.chat_deployment}")

            # Call Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer = response.choices[0].message.content
            logging.info(f"✅ Generated response ({len(answer)} chars)")

            result = {
                "answer": answer,
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
            logging.error(f"❌ Error generating response: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise

    def _build_system_prompt(self, course_name: str) -> str:
        """Build the system prompt for the AI assistant"""
        return f"""You are a helpful AI teaching assistant for the {course_name} course.

Your responsibilities:
1. Answer student questions accurately using the provided context from course materials
2. If the context doesn't contain enough information, say so honestly
3. Cite sources when possible using the [Source N] notation
4. Be clear, concise, and educational
5. If asked about topics outside the course materials, politely redirect to course content

Always prioritize accuracy over making assumptions."""

    def _build_user_message(self, question: str, context: str) -> str:
        """Build the user message with question and context"""
        return f"""Context from course materials:

{context}

---

Student Question: {question}

Please provide a helpful answer based on the context above. 
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