import argparse
import os
from docqa import DocQA


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--documents", type=str)
    parser.add_argument("--api_key", type=str)

    args = parser.parse_args().__dict__
    documents = args.pop("documents")
    os.environ["OPENAI_API_KEY"] = args.pop("api_key")
    docqa = DocQA(documents=documents)

    while True:
        question = input(
            'Enter you question; press enter to end and type "reset" to reset chat: '
        )
        if question.lower() == "":
            break
        else:
            print(docqa.answer_query(question)["choices"][0]["message"]["content"])


if __name__ == "__main__":
    cli()
