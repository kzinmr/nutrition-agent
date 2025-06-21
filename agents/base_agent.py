import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()


class ModelProvider(Enum):
    OPENAI = "openai"


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
    arguments: dict[str, Any]
    id: str


@dataclass
class ToolResult:
    tool_call_id: str
    output: Any
    error: str | None = None


class BaseAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools: dict[str, dict[str, Any]] = {}
        self.messages: list[dict[str, Any]] = []
        self.iteration_count = 0

        # Load prompts
        prompts_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
        with open(prompts_path) as f:
            self.prompts = yaml.safe_load(f)

        # Initialize LLM client
        if config.model_provider == ModelProvider.OPENAI:
            self.client: AsyncOpenAI = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported model provider: {config.model_provider}")

    def register_tool(
        self,
        name: str,
        func: Callable[..., Any],
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Register a tool that the agent can use."""
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters,
        }

    def format_tools_for_openai(self) -> list[dict[str, Any]]:
        """Format tools for OpenAI API."""
        tools = []
        for name, tool_info in self.tools.items():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool_info["description"],
                        "parameters": tool_info["parameters"],
                    },
                }
            )
        return tools

    async def _get_llm_response(self) -> dict[str, Any]:
        """Get response from LLM with tool calling support."""
        if self.config.model_provider == ModelProvider.OPENAI:
            return await self._get_openai_response()
        else:
            raise ValueError(
                f"Unsupported model provider: {self.config.model_provider}"
            )

    async def _get_openai_response(self) -> dict[str, Any]:
        """Get response from OpenAI."""
        try:
            assert isinstance(self.client, AsyncOpenAI)
            response = await self.client.chat.completions.create(  # type: ignore[call-overload]
                model=self.config.model_name,
                messages=self.messages,
                tools=self.format_tools_for_openai(),
                tool_choice="auto",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            message = response.choices[0].message

            # Handle tool calls
            if message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            name=tc.function.name,
                            arguments=json.loads(tc.function.arguments),
                            id=tc.id,
                        )
                    )
                return {
                    "content": message.content,
                    "tool_calls": tool_calls,
                    "raw_message": message.model_dump(),
                }
            else:
                return {
                    "content": message.content,
                    "tool_calls": None,
                    "raw_message": message.model_dump(),
                }

        except Exception as e:
            console.print(f"[red]Error getting OpenAI response: {e}[/red]")
            raise

    async def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls in parallel."""
        tasks = []
        for tool_call in tool_calls:
            if tool_call.name not in self.tools:
                tasks.append(
                    self._create_error_result(
                        tool_call.id, f"Unknown tool: {tool_call.name}"
                    )
                )
            else:
                tasks.append(self._execute_single_tool(tool_call))

        results = await asyncio.gather(*tasks)
        return results

    async def _execute_single_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        try:
            tool_func = self.tools[tool_call.name]["function"]
            result = await tool_func(**tool_call.arguments)
            return ToolResult(tool_call_id=tool_call.id, output=result)
        except Exception as e:
            return ToolResult(tool_call_id=tool_call.id, output=None, error=str(e))

    async def _create_error_result(self, tool_call_id: str, error: str) -> ToolResult:
        """Create an error result for a tool call."""
        return ToolResult(tool_call_id=tool_call_id, output=None, error=error)

    def _add_tool_results_to_messages(self, tool_results: list[ToolResult]) -> None:
        """Add tool results to message history."""
        if self.config.model_provider == ModelProvider.OPENAI:
            for result in tool_results:
                if result.error:
                    content = f"Error: {result.error}"
                else:
                    content = json.dumps(result.output)

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": content,
                    }
                )

    def _display_response(
        self, content: str, tool_calls: list[ToolCall] | None = None
    ) -> None:
        """Display agent response in a formatted way."""
        if tool_calls:
            console.print(Panel("🔧 [bold yellow]Executing Tools[/bold yellow]"))
            for tc in tool_calls:
                console.print(f"  • {tc.name}({json.dumps(tc.arguments, indent=2)})")

        if content:
            console.print(
                Panel(content, title="🤖 [bold green]Agent Response[/bold green]")
            )

    def _display_tool_results(self, results: list[ToolResult]) -> None:
        """Display tool execution results."""
        table = Table(title="Tool Results")
        table.add_column("Tool", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Output", style="white")

        for result in results:
            status = "✓ Success" if not result.error else "✗ Error"
            output = (
                str(result.output)[:100] + "..." if not result.error else result.error
            )
            table.add_row(result.tool_call_id[:8], status, output)

        console.print(table)

    async def run(self, user_input: str) -> str:
        """Run the agent with user input."""
        # Initialize messages with system prompt
        self.messages = [
            {"role": "system", "content": self.prompts["system_prompt"]},
            {"role": "user", "content": user_input},
        ]

        self.iteration_count = 0
        final_response = ""

        while self.iteration_count < self.config.max_iterations:
            self.iteration_count += 1
            console.print(
                f"\n[dim]Iteration {self.iteration_count}/{self.config.max_iterations}[/dim]"
            )

            # Get LLM response
            response = await self._get_llm_response()

            # Add assistant message to history
            # if self.config.model_provider == ModelProvider.OPENAI:
            self.messages.append(response["raw_message"])

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
