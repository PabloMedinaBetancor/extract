"""Review the words and numbers of a saved reading in code: save, load, decide, export.

This is the script of docs/how-to/review-doubtful-words.md. Run it from the repository folder:

    .venv/bin/python examples/review_words.py

It reads tests/fixtures/mismatch.pdf, saves the reading, loads it back as a later process would, and applies the
decisions in DECISIONS. A real tool would take them from a person who has looked at the words.
"""

from pathlib import Path

import extract

SOURCE = "tests/fixtures/mismatch.pdf"
SAVED = "work/mismatch.extract.json"
DECISIONS = {"p1w11": "accept", "p1w13": "accept", "p1w4": "accept", "p1w75": "accept"}


def main():
    options = extract.Options(work_dir="work")
    doc = extract.read(SOURCE, options=options)
    Path(SAVED).parent.mkdir(exist_ok=True)
    doc.save(SAVED)
    print("read:", len(doc.doubtful()), "words to review; saved", SAVED)

    doc = extract.load(SAVED, source=SOURCE, options=options)
    print("loaded:", doc.source, "-", len(doc.doubtful()), "words to review")

    crops = Path("work/crops")
    crops.mkdir(exist_ok=True)
    for word in doc.doubtful():
        (crops / f"{word.id}.png").write_bytes(doc.word_crop(word.id))
        action = DECISIONS.get(word.id)
        print(word.id, word.text, "->", action)
        if action == "accept":
            doc.accept(word.id)

    for number in doc.numbers:
        if number.status == "mismatch" and len(number.word_ids) == 1:
            word_id = number.word_ids[0]
            if DECISIONS.get(word_id) == "accept":
                doc.accept(word_id)
                print(word_id, number.text, "->", doc.word(word_id).text, doc.word(word_id).review)

    left = [number.text for number in doc.numbers if number.status == "mismatch"]
    print("to review:", len(doc.doubtful()), "- mismatches:", left)
    print([path.name for path in doc.export("out/", ("json",))])


if __name__ == "__main__":
    main()
