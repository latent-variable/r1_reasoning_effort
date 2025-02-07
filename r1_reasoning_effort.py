"""
title: R1 Reasoning Effort
author: latent-variable
github: https://github.com/latent-variable/r1_reasoning_effort
open-webui: https://openwebui.com/f/latentvariable/r1_reasoning_effort/
Set up instructions: https://o1-at-home.hashnode.dev/run-o1-at-home-privately-think-respond-pipe-tutorial-with-open-webui-ollama
version: 0.3.0
description: Multi-API reasoning effort pipeline for models like deepseek-r1 models with OpenAI/Ollama support.
Directly compatible with build in reasoning formater 
Compatible: open-webui v0.5.x

# Acknowledgments
https://github.com/qunash/r1-overthinker for the idea and original code
"""
import os 
import re
import json
import random
from time import time
from typing import Dict, List, Optional, Callable, Awaitable, Any, AsyncGenerator
import asyncio
from pydantic import BaseModel, Field
from dataclasses import dataclass
from fastapi import Request
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
            default=1, ge=1, le=100,
            description="Number of reasoning iterations 1-100"
        )
        START_THINK_TOKEN: str = Field(
            default="<think>",
            description="Token indicating reasoning phase started"
        )
        END_THINK_TOKEN: str = Field(
            default="</think>",
            description="Token indicating reasoning phase end"
        )
        TRACK_REASONING_EFFORT: bool = Field(
            default=False,
            description="Include the start of each reasoning effort iteration in the responce"
        )
    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.__user__ = None
        self.__request__ = None
        self._reset_state()
        self.THINK_XML_TAG = os.getenv("THINK_XML_TAG", "thinking")
        self.replacement_prompts = [ 
                "\nHold on - let me think deeper about this ", 
                "\nPerhaps a deeper perspective would help ",
                "\nLet me think ",
                "\nLet's take a step back and reconsider ",
                "\nLet me ponder this for a moment ",
                "\nI need a moment to reflect on this ",
                "\nLet’s explore this from another angle ",
                "\nThis requires a bit more thought ",
                "\nI should analyze this further ",
                "\nLet me reconsider this from a different perspective ",
                "\nI might need to rethink this ",
                "\nPerhaps there's a more nuanced way to approach this ",
                "\nLet's pause and reflect on this more deeply ",
                "\nI should take a closer look at this ",
                "\nA moment of deeper thought might help "
            ]

    def _reset_state(self):
        self.current_effort = 0
        self.generated_thinking_tokens = 0
        self.swap_count = 0
        self.buffer = ""

    def pipes(self):
        name = f"Reasoning_Effort_{self.valves.REASONING_EFFORT}/{self.valves.REASONING_MODEL}"
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
        original_messages = messages.copy()
        self.buffer = ""  # Reset buffer for new response
        self.generated_thinking_tokens =0
        while self.current_effort < self.valves.REASONING_EFFORT:
            
            try:

                status = f"Reasoning Effort step: {self.current_effort+1}"
                await self.emit_status(status, __event_emitter__, done=False)
         
                # Create fresh copy of messages for this attempt
                current_messages = original_messages.copy()
                if self.current_effort > 0:
                    # Add previous iteration's final response as context
                    extension = random.choice(self.replacement_prompts)
                    self.buffer += extension
                    current_messages.append({"role": "assistant", "content": self.buffer })
                    yield extension
                    

                response = await self.get_response(
                    model=self.valves.REASONING_MODEL,
                    messages=current_messages,
                    stream=True
                )
                found_end=False
                async for content in self._handle_api_stream(response):
                    # Check for end token in the accumulated buffer
                    if (self.valves.END_THINK_TOKEN.strip()  not in content.strip()) and not found_end:
                        self.generated_thinking_tokens += 1
                        yield content

                        # Print start of Reasoning Effort
                        if (self.valves.START_THINK_TOKEN.strip() in content.strip()) and self.valves.TRACK_REASONING_EFFORT:
                            yield f"Reasoning Effort-{self.current_effort+1}\n"
                        self.buffer += content
                    else:
                        found_end =True
                        self.current_effort += 1
                        if (self.current_effort < self.valves.REASONING_EFFORT):
                            if self.valves.TRACK_REASONING_EFFORT:
                                yield f"\nReasoning Effort-{self.current_effort+1}\n"
                            break 
                        else:
                            status = f"Reasoning Complete - Outputing Final Response"
                            await self.emit_status(status, __event_emitter__, done=False)
                            yield content

            except Exception as e:
                status = f"Reasoning error: {str(e)}"
                await self.emit_status(status, __event_emitter__, done=False)
                break
            

    async def pipe(self, body: dict, __user__: dict, __event_emitter__, __request__: Request, __task__=None):
        self.__user__ = User(**__user__)
        self.__request__ = __request__
        self._reset_state()
        
        if __task__ is not None:
            yield body["messages"][:20]
        
        try:
            async for content in self._generate_reasoning(body["messages"], __event_emitter__):
                yield content

            status = f"Completed with {self.generated_thinking_tokens} reasoning tokens"
            await self.emit_status(status, __event_emitter__, done=True)
        
        except Exception as e:
            status = f"Pipeline error: {str(e)}"
            await self.emit_status(status, __event_emitter__, done=True)
        
        yield""
    
    async def emit_status(self, status, __event_emitter__, done=False):
        await __event_emitter__({
                "type": "status",
                "data": {"description": status, "done": done}
            })
