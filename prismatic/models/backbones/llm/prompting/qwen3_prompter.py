"""
qwen3_prompter.py

PromptBuilder implementation that mirrors the chat template shipped with the
public Qwen3 checkpoints. Compared to Qwen 2.5 the model now supports
"thinking" mode by default, which wraps assistant responses in
``<think>...</think>`` blocks before the final answer. The builder keeps track
of that state so we always emit the correct assistant preamble when we ask the
model to generate the next turn.
"""

from typing import Optional

from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder

# Default System Prompt for Qwen 3 Models
SYS_PROMPTS = {
    "prismatic": (
        "You are a helpful assistant."
    ),
}


class Qwen3ChatPromptBuilder(PromptBuilder):
    def __init__(
        self,
        model_family: str,
        system_prompt: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> None:
        super().__init__(model_family, system_prompt)

        # Populate default system prompt if one is not provided.
        self.system_prompt = SYS_PROMPTS.get(self.model_family, "") if system_prompt is None else system_prompt

        # Qwen 3 sticks with ChatML style tags but introduces a thinking preamble.
        self.im_start, self.im_end = "<|im_start|>", "<|im_end|>"
        self.think_start, self.think_end = "<think>", "</think>"

        self.enable_thinking = enable_thinking

        # Lambdas keep the wrapping logic concise and uniform with existing builders.
        self.wrap_system = lambda msg: f"{self.im_start}system\n{msg}{self.im_end}\n" if msg else ""
        self.wrap_human = lambda msg: f"{self.im_start}user\n{msg}{self.im_end}\n"

        # === Builder state ===
        self.prompt, self.turn_count = "", 0
        self._pending_assistant_prefix: str = ""

    # --- Public helpers -------------------------------------------------
    def set_thinking_mode(self, enabled: bool) -> None:
        """Override the thinking mode for future turns.

        If we already primed a generation prefix for the assistant we rebuild it
        so the subsequent call to ``get_prompt`` reflects the new mode.
        """

        if self.enable_thinking == enabled:
            return

        self.enable_thinking = enabled

        if self._pending_assistant_prefix and self.prompt.endswith(self._pending_assistant_prefix):
            self.prompt = self.prompt[: -len(self._pending_assistant_prefix)]
            self._pending_assistant_prefix = self._assistant_prefix(enabled)
            self.prompt += self._pending_assistant_prefix

    # --- Core PromptBuilder API ----------------------------------------
    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")

        cleaned_message = self._sanitize_message(message)

        if role == "human":
            next_mode = self._derive_next_mode(cleaned_message)
            assistant_prefix = self._assistant_prefix(next_mode)

            if self.turn_count == 0:
                sys_message = self.wrap_system(self.system_prompt)
                human_message = self.wrap_human(cleaned_message)
                wrapped_message = sys_message + human_message + assistant_prefix
            else:
                human_message = self.wrap_human(cleaned_message)
                wrapped_message = human_message + assistant_prefix

            self.enable_thinking = next_mode
            self._pending_assistant_prefix = assistant_prefix

        else:  # role == "gpt"
            if self._pending_assistant_prefix and self.prompt.endswith(self._pending_assistant_prefix):
                self.prompt = self.prompt[: -len(self._pending_assistant_prefix)]
            self._pending_assistant_prefix = ""

            wrapped_message = self._wrap_assistant_response(cleaned_message)

        self.prompt += wrapped_message
        self.turn_count += 1
        return wrapped_message

    def get_potential_prompt(self, message: str) -> str:
        # Assumes that it's the user's (human's) turn.
        prompt_copy = str(self.prompt)
        cleaned_message = self._sanitize_message(message)
        next_mode = self._derive_next_mode(cleaned_message)
        assistant_prefix = self._assistant_prefix(next_mode)

        if self.turn_count == 0:
            sys_message = self.wrap_system(self.system_prompt)
            human_message = self.wrap_human(cleaned_message)
            prompt_copy += sys_message + human_message + assistant_prefix
        else:
            human_message = self.wrap_human(cleaned_message)
            prompt_copy += human_message + assistant_prefix

        return prompt_copy.rstrip()

    def get_prompt(self) -> str:
        return self.prompt.rstrip()

    # --- Internal helpers -----------------------------------------------
    def _assistant_prefix(self, thinking_mode: Optional[bool] = None) -> str:
        enabled = self.enable_thinking if thinking_mode is None else thinking_mode
        prefix = f"{self.im_start}assistant\n"
        if enabled:
            prefix += f"{self.think_start}\n\n{self.think_end}\n\n"
        return prefix

    def _wrap_assistant_response(self, message: str) -> str:
        content = message.rstrip()
        response = f"{self.im_start}assistant\n{content}"
        if not content.endswith(self.im_end):
            response += f"{self.im_end}\n"
        elif not response.endswith("\n"):
            response += "\n"
        return response

    def _sanitize_message(self, message: str) -> str:
        return message.replace("<image>", "").strip()

    def _derive_next_mode(self, message: str) -> bool:
        stripped = message.rstrip()
        if stripped.endswith("/no_think"):
            return False
        if stripped.endswith("/think"):
            return True
        return self.enable_thinking
