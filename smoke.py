"""Exercise both tasks with the actual pretrained model and save the results."""

import argparse
import json

from backend import GenerationOptions, ModelManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--enhance", action="store_true", help="Rewrite the editing prompt first")
    args = parser.parse_args()
    manager = ModelManager()
    try:
        image, _, metadata = manager.generate(
            [], GenerationOptions(
                "A red ceramic teapot on a white table, studio photograph.",
                args.size, args.size, args.steps, 42,
            ),
        )
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
        edit_prompt = "Change the teapot color to blue. Keep the table and composition unchanged."
        if args.enhance:
            edit_prompt = manager.rewrite_prompt([image], edit_prompt)
            print(f"Rewritten prompt: {edit_prompt}", flush=True)
        edited, _, metadata = manager.generate(
            [image], GenerationOptions(
                edit_prompt,
                args.size, args.size, args.steps, 42,
            ),
        )
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
        print(f"PASS: text-to-image={image}\nPASS: image-to-image={edited}", flush=True)
    finally:
        manager.unload()


if __name__ == "__main__":
    main()
