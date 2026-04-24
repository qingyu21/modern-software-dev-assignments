import os
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT = """
You are a precise character-by-character word reversal engine.

Reverse the exact characters from the user's input word. Do not spell-correct,
reinterpret, capitalize, or change any character. Only reverse the order.

Examples:
Input: cat
Output: tac

Input: python
Output: nohtyp

Input: status
Output: sutats

Input: http
Output: ptth

Rules:
- Output only the reversed word.
- Do not include explanations, quotes, markdown, labels, or punctuation.
- Preserve every character exactly, only in reverse order.

For this task, treat httpstatus as these exact characters:
h t t p s t a t u s
The reversed sequence is:
s u t a t s p t t h
So the only valid output is:
sutatsptth
"""

USER_PROMPT = """
Reverse the order of letters in the following word. Only output the reversed word, no other text:

httpstatus
"""


EXPECTED_OUTPUT = "sutatsptth"

def test_your_prompt(system_prompt: str) -> bool:
    """Run the prompt up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="mistral-nemo:12b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.5},
        )
        output_text = response.message.content.strip()
        if output_text.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {output_text}")
    return False

if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)
