"""
qwen25_prompter.py

Defines a PromptBuilder for building Qwen 2.5 Chat Prompts.

Reference: https://huggingface.co/Qwen/Qwen2-7B-Instruct
"""

from typing import Optional

from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder

# Default System Prompt for Qwen 2.5 Models
SYS_PROMPTS = {
    "prismatic": (
        "You are a helpful assistant."
    ),
}


class Qwen25ChatPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        # Get system prompt; use default if not provided
        self.system_prompt = SYS_PROMPTS.get(self.model_family, "") if system_prompt is None else system_prompt

        # Qwen 2.5 uses specific tokens for conversation roles
        self.im_start, self.im_end = "<|im_start|>", "<|im_end|>"

        # Get role-specific "wrap" functions
        self.wrap_system = lambda msg: f"{self.im_start}system\n{msg}{self.im_end}\n"
        self.wrap_human = lambda msg: f"{self.im_start}user\n{msg}{self.im_end}\n"
        self.wrap_gpt = lambda msg: f"{self.im_start}assistant\n{msg if msg != '' else ''}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        # Special Handling for the first turn to include the system prompt
        if self.turn_count == 0:
            sys_message = self.wrap_system(self.system_prompt)
            human_message = self.wrap_human(message)
            wrapped_message = sys_message + human_message + self.wrap_gpt("")
        elif (self.turn_count % 2) == 0:
            # Add Human turn, followed by the start of the assistant's turn
            human_message = self.wrap_human(message)
            wrapped_message = human_message + self.wrap_gpt("")
        else:
            # Complete the assistant's turn
            gpt_message = message + f"{self.im_end}\n"
            # Remove the placeholder and add the actual message
            self.prompt = self.prompt.rstrip(self.wrap_gpt(""))
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> str:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        if self.turn_count == 0:
            sys_message = self.wrap_system(self.system_prompt)
            human_message = self.wrap_human(message)
            prompt_copy += sys_message + human_message + self.wrap_gpt("")
        else:
            human_message = self.wrap_human(message)
            prompt_copy += human_message + self.wrap_gpt("")

        return prompt_copy.rstrip()

    def get_prompt(self) -> str:
        return self.prompt.rstrip()