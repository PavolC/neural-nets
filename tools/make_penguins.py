"""Build public/data/penguins.json.gz, the course's second dataset.

Chapter 10 is about applying the network to a problem of the learner's own, and
MNIST is useless for teaching that: its pixels arrive already scaled to 0 and 1,
already numeric, already complete, already split. This one arrives the way data
actually does.

    344 penguins, three species, and:
      * four measurements on wildly different scales (body mass in grams runs
        about 4,200; bill depth in millimetres runs about 17, a factor of 245),
        which is what makes Chapter 7's initialization argument break and makes
        input scaling the chapter's central lesson
      * two categorical columns (island, sex) that have to be encoded
      * holes: two rows with no measurements at all, eleven with no sex
      * classes in unequal numbers (152 / 124 / 68), so accuracy has to be read
        against a majority-class baseline rather than against zero

The file is written RAW on purpose: numbers as numbers, categories as strings,
missing values as null, rows in source order, nothing scaled and nothing split.
Preparing it is the exercise.

Data: the Palmer Station Antarctica LTER penguin study, collected by Dr Kristen
Gorman and published as the palmerpenguins R package by Allison Horst, Alison
Hill and Kristen Gorman, released CC0 (public domain). Please keep the citation
in the chapter and the README.

Pure standard library, needs network access:

    python3 tools/make_penguins.py
"""

import gzip
import json
import pathlib
import urllib.request

SOURCE = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
    "main/inst/extdata/penguins.csv"
)
OUT = pathlib.Path(__file__).resolve().parent.parent / "public" / "data" / "penguins.json.gz"

# The columns the course keeps, in the order the bundled file uses. `year` is
# dropped: it is a fact about the survey rather than about the penguin, and a
# learner who feeds it in is being taught the wrong thing.
COLUMNS = [
    "species",
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]
NUMERIC = {"bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"}


def parse_csv(text):
    """Minimal CSV reader: this file has no quoted fields or embedded commas."""
    lines = [line for line in text.splitlines() if line.strip()]
    header = lines[0].split(",")
    for line in lines[1:]:
        yield dict(zip(header, line.split(",")))


def clean(value, column):
    if value in ("NA", "", None):
        return None
    if column in NUMERIC:
        number = float(value)
        # Body mass and flipper length are whole numbers in the source; keep
        # them that way so the bundled file reads like the measurements do.
        return int(number) if number.is_integer() else number
    return value


def main():
    with urllib.request.urlopen(SOURCE) as resp:
        text = resp.read().decode("utf-8")

    rows = [[clean(r[c], c) for c in COLUMNS] for r in parse_csv(text)]

    payload = {
        "name": "Palmer Archipelago penguins",
        "source": SOURCE,
        "citation": (
            "Horst AM, Hill AP, Gorman KB (2020). palmerpenguins: Palmer "
            "Archipelago (Antarctica) penguin data. Data collected by Dr Kristen "
            "Gorman, Palmer Station Antarctica LTER. Released CC0."
        ),
        "license": "CC0 1.0 (public domain dedication)",
        "columns": COLUMNS,
        "rows": rows,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    # mtime=0 so rebuilding the file from the same source produces the same
    # bytes, the way tools/make_mnist_subset.py does.
    with gzip.GzipFile(OUT, "wb", compresslevel=9, mtime=0) as f:
        f.write(raw)

    missing = sum(1 for row in rows if any(v is None for v in row))
    species = {}
    for row in rows:
        species[row[0]] = species.get(row[0], 0) + 1
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes gzipped, {len(raw):,} raw)")
    print(f"  {len(rows)} rows, {len(COLUMNS)} columns, {missing} with a hole in them")
    print("  species: " + ", ".join(f"{k} {v}" for k, v in sorted(species.items())))


if __name__ == "__main__":
    main()
