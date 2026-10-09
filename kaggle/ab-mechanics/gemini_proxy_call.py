"""Isolated direct model call; stdin/stdout JSON, no workspace-agent context."""
import json
import sys

from google import genai
from google.genai import types


def main():
    prompt = json.load(sys.stdin)['prompt']
    client = genai.Client()
    answer = client.models.generate_content(
        model='gemini-2.5-flash', contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction='You are the game agent in a bounded local test. Return one executable Python code block only.',
            temperature=0, max_output_tokens=2500,
        ),
    )
    print(json.dumps({'text': answer.text or '',
                      'usage': answer.usage_metadata.model_dump() if answer.usage_metadata else None,
                      'finish_reason': str(answer.candidates[0].finish_reason) if answer.candidates else None}))


if __name__ == '__main__':
    main()
