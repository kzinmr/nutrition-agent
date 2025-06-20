import asyncio
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import yaml
from pathlib import Path

import openai
from anthropic import AsyncAnthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import os
from dotenv import load_dotenv

load_dotenv()

console = Console()


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class AgentConfig:
    model_provider: ModelProvider
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 4000
    max_iterations: int = 10
    

@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    id: str


@dataclass
class ToolResult:
    tool_call_id: str
    output: Any
    error: Optional[str] = None


class BaseAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools: Dict[str, Callable] = {}
        self.messages: List[Dict[str, Any]] = []
        self.iteration_count = 0
        
        # Load prompts
        prompts_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
        with open(prompts_path, 'r') as f:
            self.prompts = yaml.safe_load(f)
        
        # Initialize LLM client
        if config.model_provider == ModelProvider.OPENAI:
            self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif config.model_provider == ModelProvider.ANTHROPIC:
            self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unsupported model provider: {config.model_provider}")
    
    def register_tool(self, name: str, func: Callable, description: str, parameters: Dict[str, Any]):
        """Register a tool that the agent can use."""
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }
    
    def _format_tools_for_openai(self) -> List[Dict[str, Any]]:
        """Format tools for OpenAI API."""
        tools = []
        for name, tool_info in self.tools.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool_info["description"],
                    "parameters": tool_info["parameters"]
                }
            })
        return tools
    
    def _format_tools_for_anthropic(self) -> List[Dict[str, Any]]:
        """Format tools for Anthropic API."""
        tools = []
        for name, tool_info in self.tools.items():
            tools.append({
                "name": name,
                "description": tool_info["description"],
                "input_schema": tool_info["parameters"]
            })
        return tools
    
    async def _get_llm_response(self) -> Dict[str, Any]:
        """Get response from LLM with tool calling support."""
        if self.config.model_provider == ModelProvider.OPENAI:
            return await self._get_openai_response()
        elif self.config.model_provider == ModelProvider.ANTHROPIC:
            return await self._get_anthropic_response()
    
    async def _get_openai_response(self) -> Dict[str, Any]:
        """Get response from OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=self.messages,
                tools=self._format_tools_for_openai(),
                tool_choice="auto",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            message = response.choices[0].message
            
            # Handle tool calls
            if message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    tool_calls.append(ToolCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                        id=tc.id
                    ))
                return {
                    "content": message.content,
                    "tool_calls": tool_calls,
                    "raw_message": message.model_dump()
                }
            else:
                return {
                    "content": message.content,
                    "tool_calls": None,
                    "raw_message": message.model_dump()
                }
                
        except Exception as e:
            console.print(f"[red]Error getting OpenAI response: {e}[/red]")
            raise
    
    async def _get_anthropic_response(self) -> Dict[str, Any]:
        """Get response from Anthropic."""
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            system_message = None
            
            for msg in self.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append(msg)
            
            response = await self.client.messages.create(
                model=self.config.model_name,
                messages=anthropic_messages,
                system=system_message,
                tools=self._format_tools_for_anthropic(),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Handle tool calls
            tool_calls = []
            for content in response.content:
                if content.type == "tool_use":
                    tool_calls.append(ToolCall(
                        name=content.name,
                        arguments=content.input,
                        id=content.id
                    ))
            
            # Get text content
            text_content = ""
            for content in response.content:
                if content.type == "text":
                    text_content += content.text
            
            return {
                "content": text_content,
                "tool_calls": tool_calls if tool_calls else None,
                "raw_message": response.model_dump()
            }
            
        except Exception as e:
            console.print(f"[red]Error getting Anthropic response: {e}[/red]")
            raise
    
    async def _execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute tool calls in parallel."""
        tasks = []
        for tool_call in tool_calls:
            if tool_call.name not in self.tools:
                tasks.append(self._create_error_result(
                    tool_call.id,
                    f"Unknown tool: {tool_call.name}"
                ))
            else:
                tasks.append(self._execute_single_tool(tool_call))
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def _execute_single_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        try:
            tool_func = self.tools[tool_call.name]["function"]
            result = await tool_func(**tool_call.arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                output=result
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                output=None,
                error=str(e)
            )
    
    async def _create_error_result(self, tool_call_id: str, error: str) -> ToolResult:
        """Create an error result for a tool call."""
        return ToolResult(
            tool_call_id=tool_call_id,
            output=None,
            error=error
        )
    
    def _add_tool_results_to_messages(self, tool_results: List[ToolResult]):
        """Add tool results to message history."""
        if self.config.model_provider == ModelProvider.OPENAI:
            for result in tool_results:
                if result.error:
                    content = f"Error: {result.error}"
                else:
                    content = json.dumps(result.output)
                
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": content
                })
        
        elif self.config.model_provider == ModelProvider.ANTHROPIC:
            tool_results_content = []
            for result in tool_results:
                if result.error:
                    content = {"error": result.error}
                else:
                    content = result.output
                
                tool_results_content.append({
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": json.dumps(content)
                })
            
            self.messages.append({
                "role": "user",
                "content": tool_results_content
            })
    
    def _display_response(self, content: str, tool_calls: Optional[List[ToolCall]] = None):
        """Display agent response in a formatted way."""
        if tool_calls:
            console.print(Panel("🔧 [bold yellow]Executing Tools[/bold yellow]"))
            for tc in tool_calls:
                console.print(f"  • {tc.name}({json.dumps(tc.arguments, indent=2)})")
        
        if content:
            console.print(Panel(content, title="🤖 [bold green]Agent Response[/bold green]"))
    
    def _display_tool_results(self, results: List[ToolResult]):
        """Display tool execution results."""
        table = Table(title="Tool Results")
        table.add_column("Tool", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Output", style="white")
        
        for result in results:
            status = "✓ Success" if not result.error else "✗ Error"
            output = str(result.output)[:100] + "..." if not result.error else result.error
            table.add_row(result.tool_call_id[:8], status, output)
        
        console.print(table)
    
    async def run(self, user_input: str) -> str:
        """Run the agent with user input."""
        # Initialize messages with system prompt
        self.messages = [
            {"role": "system", "content": self.prompts["system_prompt"]},
            {"role": "user", "content": user_input}
        ]
        
        self.iteration_count = 0
        final_response = ""
        
        while self.iteration_count < self.config.max_iterations:
            self.iteration_count += 1
            console.print(f"\n[dim]Iteration {self.iteration_count}/{self.config.max_iterations}[/dim]")
            
            # Get LLM response
            response = await self._get_llm_response()
            
            # Add assistant message to history
            if self.config.model_provider == ModelProvider.OPENAI:
                self.messages.append(response["raw_message"])
            elif self.config.model_provider == ModelProvider.ANTHROPIC:
                self.messages.append({
                    "role": "assistant",
                    "content": response["raw_message"]["content"]
                })
            
            # Display response
            self._display_response(response["content"], response["tool_calls"])
            
            # If there are tool calls, execute them
            if response["tool_calls"]:
                tool_results = await self._execute_tool_calls(response["tool_calls"])
                self._display_tool_results(tool_results)
                self._add_tool_results_to_messages(tool_results)
            else:
                # No more tool calls, we have the final response
                final_response = response["content"]
                break
        
        if self.iteration_count >= self.config.max_iterations:
            console.print("[red]Maximum iterations reached![/red]")
        
        return final_response