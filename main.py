import argparse
from lairgpt.models import PAGnol


def main(args):
    size = args.size
    assert size in ["small", "large"], "Unrecognized model size!"
    text = args.text

    if size == "small":
        pagnol = PAGnol.small()
    if size == "large":
        pagnol = PAGnol.large()

    n_steps = 5
    for _ in range(n_steps):
        generated_text = pagnol(text, mode="nucleus")[0]
        text = f"{text} {generated_text}"

    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run LightOn's French-GPT")
    parser.add_argument("--size", type=str, help="model size")
    parser.add_argument("--text", type=str, help="input text")
    args = parser.parse_args()
    main(args)
