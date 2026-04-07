#!/usr/bin/env python3
import sys
import os
import json

def read_prompt():
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    try:
        data = sys.stdin.read()
        return data.strip()
    except Exception:
        return ''

def main():
    prompt = read_prompt()
    if not prompt:
        print(json.dumps({"reply":"No prompt provided","actions":[]}))
        return
    try:
        # Try to use gpt4all python package
        from gpt4all import GPT4All
        model_path = os.environ.get('GPT4ALL_MODEL_PATH', './models/gpt4all-lora-quantized.bin')
        gpt = GPT4All(model=model_path)
        # generate text
        out = gpt.generate(prompt, max_tokens=200)
        # ensure string
        if isinstance(out, (list, tuple)):
            out = ''.join(out)
        if not isinstance(out, str):
            out = str(out)
        # print JSON wrapper
        print(json.dumps({"reply": out, "actions": []}, ensure_ascii=False))
    except Exception as e:
        # If gpt4all not available or fails, return a simulated structured response
        err = str(e)
        fallback_reply = "Simulated response: no model available or error: " + err
        fallback_actions = []
        # simple heuristic: if user asked to create a box include an action
        low = prompt.lower()
        if any(w in low for w in ['create','draw','make','box','cube','sphere','cylinder']):
            fallback_actions = [{"type":"create","command":"create box size 1x1x1 at 0 0 0"}]
            fallback_reply = "Simulated: plan para crear geometría incluido."
        print(json.dumps({"reply": fallback_reply, "actions": fallback_actions}, ensure_ascii=False))

if __name__ == '__main__':
    main()
