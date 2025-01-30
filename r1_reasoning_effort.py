"""
title: R1 Reasoning Effort
author: latent-variable
github: https://github.com/latent-variable/r1_reasoning_effort
open-webui: your_webui_link
version: 0.3
description: Multi-API reasoning pipeline with OpenAI/Ollama support.
"""

import json
import random
from time import time
from typing import Dict, List, Optional, Callable, Awaitable, Any, AsyncGenerator
import asyncio
from pydantic import BaseModel, Field
from dataclasses import dataclass
from fastapi import Request
from open_webui.utils.misc import get_last_user_message
from open_webui.routers.ollama import generate_chat_completion as ollama_completion
from open_webui.routers.openai import generate_chat_completion as openai_completion
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

@dataclass
class User:
    id: str
    email: str
    name: str
    role: str

class Pipe:
    class Valves(BaseModel):
        # API Selection
        USE_OPENAI_REASONING: bool = Field(
            default=False,
            description="Use an OpenAI API for instead of ollama "
        )
        REASONING_MODEL: str = Field(
            default="deepseek-r1:8b",
            description="Model for reasoning phase (Ollama name or OpenAI ID)"
        )
        # Reasoning Parameters
        REASONING_EFFORT: int = Field(
            default=1, ge=1, le=5,
            description="Number of reasoning iterations"
        )
        END_THINK_TOKEN: str = Field(
            default="\n</think>\n",
            description="Token indicating reasoning phase end"
        )
        REPLACEMENT_PROMPTS: str = Field(
            default="""Let me reconsider...\nAnother perspective...""",
            description="Prompts for extended reasoning (one per line)"
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.__user__ = None
        self.__request__ = None
        self._reset_state()
        self.replacement_prompts = [ "\nHold on - let me think deeper about this ", 
                                     "\nPerhaps a deeper perspective would help ",
                                     "\nLet me think "
                                    ]

    def _reset_state(self):
        self.current_effort = 0
        self.generated_tokens = 0
        self.swap_count = 0
        self.buffer = ""

    def pipes(self):
        name = f"R1 Reasoning Effort: {self.valves.REASONING_MODEL} Effort: {self.valves.REASONING_EFFORT}"
        return [{"name": name, "id": name}]

    async def get_response(self, model: str, messages: List[Dict], stream: bool):
        """Unified API request handler"""
        use_openai = self.valves.USE_OPENAI_REASONING 
        
        try:
            if use_openai:
                response = await openai_completion(
                    self.__request__,
                    {"model": model, "messages": messages, "stream": stream},
                    user=self.__user__
                )
            else:
                response = await ollama_completion(
                    self.__request__,
                    {"model": model, "messages": messages, "stream": stream},
                    user=self.__user__
                )
            return response
        except Exception as e:
            logger.error(f"API Error ({'OpenAI' if use_openai else 'Ollama'}): {str(e)}")
            raise

    async def _handle_api_stream(self, response):
        """Unified stream handler for both APIs"""
        buffer = ""
        async for chunk in response.body_iterator:
            if self.valves.USE_OPENAI_REASONING:
                # Handle OpenAI's streaming format
                if chunk.startswith("data: "):
                    try:
                        data = json.loads(chunk[6:])
                        if 'choices' in data and len(data['choices']) > 0:
                            content = data['choices'][0]['delta'].get('content', '')
                            buffer += content
                            yield content
                    except json.JSONDecodeError:
                        continue
            else:
                # Handle Ollama's format
                buffer += chunk.decode()
                if '\n' in buffer:
                    lines = buffer.split('\n')
                    for line in lines[:-1]:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                yield data.get("message", {}).get("content", "")
                            except json.JSONDecodeError:
                                continue
                    buffer = lines[-1]

    async def _generate_reasoning(self, messages: List[Dict], __event_emitter__):
        """Core reasoning pipeline with token interception"""
        attempt = 0
        original_messages = messages.copy()
        self.buffer = ""  # Reset buffer for new response
        while self.current_effort < self.valves.REASONING_EFFORT:
            attempt += 1
            try:
                # Create fresh copy of messages for this attempt
                current_messages = original_messages.copy()
                if self.current_effort > 0:
                    # Add previous iteration's final response as context
                    extension = random.choice(self.replacement_prompts)
                    content = self.buffer + extension
                    current_messages.append({"role": "assistant", "content": content })
                    yield extension

                response = await self.get_response(
                    model=self.valves.REASONING_MODEL,
                    messages=current_messages,
                    stream=True
                )

                async for content in self._handle_api_stream(response):
                    
                    # Check for end token in the accumulated buffer
                    if self.valves.END_THINK_TOKEN  not in content:
                        self.buffer += content
                        self.generated_tokens += len(content)
                        yield content
                    else:
                        self.current_effort += 1
                        if self.current_effort < self.valves.REASONING_EFFORT:
                            # 
                            yield "<think>\n"
                            break
                        else:
                            yield content

            except Exception as e:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Reasoning error: {str(e)}", "done": False}
                })
                break
            

    async def pipe(self, body: dict, __user__: dict, __event_emitter__, __request__: Request, __task__=None):
        self.__user__ = User(**__user__)
        self.__request__ = __request__
        self._reset_state()
        
        if __task__ is not None:
            return body["messages"][:20]
        
        try:
            async for content in self._generate_reasoning(body["messages"], __event_emitter__):
                
                await __event_emitter__({
                    "type": "message",
                    "data": {
                        "content": content,
                        "role": "assistant-reasoning",
                        "metadata": {
                            "effort": self.current_effort,
                            "total_tokens": self.generated_tokens
                        }
                    }
                })

        except Exception as e:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Pipeline error: {str(e)}", "done": True}
            })
        
        return ""
    