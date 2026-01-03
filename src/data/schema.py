from dataclasses import dataclass

@dataclass
class Example:
    instruction: str
    input: str
    output: str

    def to_prompt(self) -> str:
        return (
            "### Instruction:\n"
            f"{self.instruction.strip()}\n\n"
            "### Input:\n"
            f"{self.input.strip()}\n\n"
            "### Response:\n"
        )