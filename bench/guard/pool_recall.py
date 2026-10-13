"""Ceilings of the candidate pool, measured on the cached pages of the v4 corpus.

For the wrong words of the development sets this counts the errors by kind, how often each candidate source holds
the truth (as written and at the level of its letters), what the v4 fixer reached, and what a join, a
segmentation or a drop could reach. It reads pages that already carry an OCR reading and never opens a sealed set.

    python3 -s -m bench.guard.pool_recall --sets train val --out bench/results/pool-recall-cached.json
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from bench.corpus.labels import Label, TruthWord
from bench.guard import artifacts, dataset
from extract.engine.cache import reading_from_dict
from extract.guard import candidates, dictionary, lexicon
from extract.guard.forms import apply_case, case_variants, normalise, split_word
from extract.types import Word

LANGUAGES = ("es", "en")
DIGIT = re.compile(r"\d")
SINGLE_LETTER_WORDS = frozenset("ayoeu")
OUTSIDE_SAMPLE = 20_000
NEIGHBOUR_RATIO = 3
V4_POOL = 20
V4_NEAR = 60
CEILING_EDITS = 3
SOURCES = ("dictionary", "fold", "punct", "page")


@dataclass
class PageCase:
    doc: str
    page: int
    band: str
    words: tuple[Word, ...]
    labels: list[Label]
    truth: list[TruthWord]


def load_cases(data_dir: Path, set_names) -> tuple[list[PageCase], Counter]:
    """The readable pages of the sets, labelled; the unreadable ones are only counted. Sealed sets are refused."""
    cases: list[PageCase] = []
    counts: Counter = Counter()
    for set_name in set_names:
        for _, record in dataset.page_records(data_dir, set_name):
            reading = reading_from_dict(record["ocr"]["reading"])
            if reading.status != "ocr":
                counts["unreadable_pages"] += 1
                continue
            truth = [TruthWord(w["id"], w["text"], tuple(w["box"])) for w in record["truth"]["words"]]
            read = [(w.id, w.text, w.box) for w in reading.words]
            alignment, _ = artifacts.align(read, truth, record.get("ungrounded", []))
            cases.append(
                PageCase(record["doc_id"], record["page"], record["band"], reading.words, alignment.labels, truth)
            )
            counts["pages"] += 1
    return cases, counts


def error_kind(label: Label, word: Word) -> str:
    """`right`, `ungrounded`, `digit`, `letters`, `accent`, `punctuation`, `case`, `merge`, `split`, `mixed` or `invented`."""
    if label.wrong is None:
        return "ungrounded"
    if not label.wrong:
        return "right"
    if label.kind != "substitution":
        return label.kind
    if DIGIT.search(word.text) or DIGIT.search(label.truth_text or ""):
        return "digit"
    return label.detail or "letters"


def v4_texts(token: str) -> list[str]:
    """What the v4 fixer proposed for a token: the 20 most frequent of the 60 nearest listed words and the letters
    under the other casings, each in the token's case and punctuation. Kept here so v4 stays reproducible."""
    parts = split_word(token)
    if parts is None:
        return []
    prefix, core, suffix = parts
    found: dict[str, float] = {}
    for language in lexicon.available(LANGUAGES):
        book = dictionary.load(language)
        for rank, _ in book.candidates(core)[:V4_NEAR]:
            word = book.words[rank]
            found[word] = max(found.get(word, 0.0), float(book.log_freq[rank]))
    texts = [prefix + apply_case(word, core) + suffix for word in sorted(found, key=lambda w: -found[w])[:V4_POOL]]
    texts.extend(prefix + variant + suffix for variant in case_variants(core) if variant != core)
    return [text for text in dict.fromkeys(texts) if text != token]


def letters(text: str) -> str:
    return "".join(c for c in normalise(text).casefold() if c.isalpha())


def reached(texts, truth: str) -> tuple[bool, bool]:
    """Whether the truth is among the texts as written, and at the level of its letters."""
    written = any(normalise(text) == truth for text in texts)
    return written, written or any(letters(text) == letters(truth) for text in texts)


def share(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def error_table(cases: list[PageCase]) -> dict:
    """Wrong words by kind (digits apart), the merges and the splits broken down by how small a repair they need."""
    kinds: Counter = Counter()
    merges: Counter = Counter()
    split_groups: Counter = Counter()
    for case in cases:
        index = {w.id: i for i, w in enumerate(case.words)}
        groups: dict[tuple, list[int]] = {}
        for word, label in zip(case.words, case.labels, strict=True):
            kind = error_kind(label, word)
            kinds[kind] += 1
            if kind == "merge":
                merges.update(_merge_repair(word.text, label.truth_text))
            elif kind == "split":
                groups.setdefault(label.truth_ids, []).append(index[label.ocr_id])
        for ids, members in groups.items():
            truth = next(w.text for w in case.truth if w.id == ids[0])
            joined = "".join(normalise(case.words[i].text) for i in sorted(members))
            split_groups["groups"] += 1
            split_groups["exact"] += joined == normalise(truth)
            split_groups["within_2_edits"] += Levenshtein.distance(joined, normalise(truth)) <= 2
    wrong = sum(v for k, v in kinds.items() if k not in ("right", "ungrounded"))
    return {
        "words": dict(sorted(kinds.items())),
        "wrong": wrong,
        "wrong_share_with_digits": share(kinds["digit"], wrong),
        "merge_repair": dict(merges),
        "split_groups": dict(split_groups),
    }


def _merge_repair(text: str, truth_text: str | None) -> Counter:
    if not truth_text:
        return Counter(tokens=1)
    pieces = truth_text.split(" ")
    glued = "".join(pieces)
    token = normalise(text)
    out = Counter(tokens=1)
    out["only_miss_the_space"] += token == glued
    out["within_2_edits_of_no_space"] += Levenshtein.distance(token, glued) <= 2
    return out


@dataclass
class Site:
    case: PageCase
    index: int
    token: str
    truth: str
    detail: str
    plain: bool
    site: bool
    texts: dict[str, list[str]]
    v4: list[str]
    capped: list[str]


def wrong_sites(cases: list[PageCase], context_rule: bool = True) -> list[Site]:
    """One entry per wrong one-to-one word without digits: what each source proposes and what v4 proposed."""
    out: list[Site] = []
    for case in cases:
        ctx = candidates.PageContext(case.words, LANGUAGES, _fragments(case.words))
        for i, (word, label) in enumerate(zip(case.words, case.labels, strict=True)):
            kind = error_kind(label, word)
            if kind not in ("letters", "accent", "punctuation", "case"):
                continue
            plain = ctx.parts[i] is not None
            site = ctx.is_site(i)
            texts = candidates.by_source(ctx, i, context_rule) if plain else {}
            capped = [c.text for c in candidates.generate(ctx, i)] if site else []
            out.append(
                Site(case, i, word.text, label.truth_text, kind, plain, site, texts, v4_texts(word.text), capped)
            )
    return out


def _fragments(words) -> set[int]:
    from extract.guard.signals import _group_by_line, _hyphen_fragments

    return _hyphen_fragments(words, _group_by_line(words))


def _union(site: Site, names) -> list[str]:
    return [text for name in names for text in site.texts.get(name, [])]


def _recall_rows(group: list[Site]) -> dict:
    """Recall, as written and by letters, of v4, of each source, of their union and of the capped list. When the words
    carry what the engine read (alternatives, masked decode, second readings) the union with them and the marginal
    gain of each of those sources come too."""
    engine = sorted({name for s in group for name in s.texts} - set(SOURCES))
    rows = {"v4": lambda s: s.v4, "union": lambda s: _union(s, SOURCES), "capped": lambda s: s.capped}
    rows.update({name: (lambda s, name=name: s.texts.get(name, [])) for name in (*SOURCES, *engine)})
    rows["dictionary+fold"] = lambda s: _union(s, ("dictionary", "fold"))
    rows["dictionary+fold+punct"] = lambda s: _union(s, ("dictionary", "fold", "punct"))
    if engine:
        rows["union_with_engine"] = lambda s: _union(s, (*SOURCES, *engine))
    out = {}
    for name, texts in rows.items():
        outcomes = [reached(texts(s), s.truth) for s in group]
        out[name] = {
            "written": share(sum(w for w, _ in outcomes), len(group)),
            "letters": share(sum(r for _, r in outcomes), len(group)),
        }
    if engine:
        full = out["union_with_engine"]["written"]
        out["marginal_gain_written"] = {
            name: round(
                full
                - share(
                    sum(reached(_union(s, [n for n in (*SOURCES, *engine) if n != name]), s.truth)[0] for s in group),
                    len(group),
                ),
                4,
            )
            for name in engine
        }
    return out


def pool_table(sites: list[Site], cases: list[PageCase]) -> dict:
    """Recall of the pool on the wrong plain one-to-one words, as written and at the level of the letters."""
    plain = [s for s in sites if s.plain]
    out: dict = {
        "wrong_one_to_one": len(sites),
        "plain": len(plain),
        "sites": sum(s.site for s in plain),
        "protected": sum(not s.site for s in plain),
    }
    names = SOURCES
    out.update(_recall_rows(plain))
    out["sites_only"] = _recall_rows([s for s in plain if s.site])
    union_by_kind = {}
    for kind in ("letters", "accent", "punctuation", "case"):
        group = [s for s in plain if s.detail == kind]
        union_by_kind[kind] = {
            "n": len(group),
            "v4": share(sum(reached(s.v4, s.truth)[0] for s in group), len(group)),
            "union": share(sum(reached(_union(s, names), s.truth)[0] for s in group), len(group)),
        }
    out["by_kind"] = union_by_kind
    out["dictionary_letters_ceiling"] = share(
        sum(
            _in_dictionary(s.truth) and Levenshtein.distance(letters(s.token), letters(s.truth)) <= CEILING_EDITS
            for s in plain
        ),
        len(plain),
    )
    out["union_written_over_ceiling"] = _ratio(out["union"]["written"], out["dictionary_letters_ceiling"])
    out["v4_written_over_ceiling"] = _ratio(out["v4"]["written"], out["dictionary_letters_ceiling"])
    out["unreached_by_cause"] = unreached_causes(sites)
    out["document_vocabulary"] = _document_vocabulary(plain, cases, names)
    out["truth_outside_dictionary"] = share(sum(not _in_dictionary(s.truth) for s in plain), len(plain))
    return out


def unreached_causes(sites: list[Site]) -> dict:
    """Why a wrong plain one-to-one word is not in the pool the chooser sees, each word under the first cause that
    applies: protected by a rule, cut by the cap of candidates, reached only by its letters (the written form is
    wrong), truth outside the dictionary, truth in the dictionary but more than three edits away, or just
    beyond the two edits the index finds."""
    plain = [s for s in sites if s.plain]
    causes: Counter = Counter()
    for s in plain:
        union = _union(s, SOURCES)
        written, by_letters = reached(union, s.truth)
        if not s.site:
            causes["protected"] += 1
        elif reached(s.capped, s.truth)[0]:
            causes["reached"] += 1
        elif written:
            causes["cut_by_the_cap"] += 1
        elif by_letters:
            causes["written_form"] += 1
        elif not _in_dictionary(s.truth):
            causes["truth_outside_dictionary"] += 1
        elif Levenshtein.distance(letters(s.token), letters(s.truth)) <= CEILING_EDITS:
            causes["within_3_edits_not_found"] += 1
        else:
            causes["beyond_3_edits"] += 1
    return {"words": len(plain), "counts": dict(causes), "shares": {k: share(v, len(plain)) for k, v in causes.items()}}


def _ratio(num: float | None, den: float | None) -> float | None:
    return round(num / den, 4) if num is not None and den else None


def _in_dictionary(truth: str) -> bool:
    core = letters(truth)
    return any(dictionary.load(language).rank_of(core) >= 0 for language in lexicon.available(LANGUAGES))


def _document_vocabulary(plain: list[Site], cases: list[PageCase], names) -> dict:
    """How often the truth is a word of another sampled page of the same document within two edits of the token
    (at least once, at least twice), and the union with the dictionary at the level of the letters."""
    per_page: dict[tuple[str, int], Counter] = {}
    for case in cases:
        per_page[(case.doc, case.page)] = Counter(
            parts[1].casefold() for w in case.words if (parts := split_word(w.text))
        )
    by_doc: dict[str, list[tuple[str, int]]] = {}
    for key in per_page:
        by_doc.setdefault(key[0], []).append(key)
    once = twice = both = 0
    for s in plain:
        elsewhere = Counter()
        for key in by_doc[s.case.doc]:
            if key != (s.case.doc, s.case.page):
                elsewhere.update(per_page[key])
        truth_letters, token_letters = letters(s.truth), letters(s.token)
        near = Levenshtein.distance(token_letters, truth_letters) <= candidates.PAGE_EDITS
        count = elsewhere[truth_letters] if near else 0
        once += count >= 1
        twice += count >= 2
        both += count >= 1 or reached(_union(s, ("dictionary", "fold")), s.truth)[1]
    n = len(plain)
    return {
        "at_least_once": share(once, n),
        "at_least_twice": share(twice, n),
        "with_dictionary_letters": share(both, n),
    }


def beyond_pool(ctx: candidates.PageContext, index: int, truth: str, alternatives=()) -> bool:
    """Whether neither the dictionary, the page vocabulary nor the given alternative texts hold the truth, as written,
    for the token at `index`: the words a reader could add something to (`recall_beyond_pool`)."""
    if ctx.parts[index] is None:
        return not any(normalise(text) == truth for text in alternatives)
    texts = [text for found in candidates.by_source(ctx, index).values() for text in found]
    return not reached([*texts, *alternatives], truth)[0]


def beyond_pool_ids(case: PageCase, alternatives=None) -> list[str]:
    """Ids of the wrong one-to-one words of a page without digits that `beyond_pool` says nobody reaches.
    `alternatives` maps a word id to the texts the engine's alternatives propose for it."""
    ctx = candidates.PageContext(case.words, LANGUAGES, _fragments(case.words))
    found = []
    for i, (word, label) in enumerate(zip(case.words, case.labels, strict=True)):
        if error_kind(label, word) in ("letters", "accent", "punctuation", "case") and beyond_pool(
            ctx, i, label.truth_text, (alternatives or {}).get(word.id, ())
        ):
            found.append(word.id)
    return found


def outside_dictionary_neighbours(cases: list[PageCase], seed: int = 1) -> dict:
    """Of the right plain words outside the dictionary (a sample), how many have a repeated word of the same page one
    edit away that is seen at least three times more often: the harm the page vocabulary could do to names."""
    pool = []
    for case in cases:
        ctx = candidates.PageContext(case.words, LANGUAGES, _fragments(case.words))
        for i, (word, label) in enumerate(zip(case.words, case.labels, strict=True)):
            if label.wrong is False and ctx.parts[i] and not _in_dictionary(word.text):
                pool.append((ctx, i))
    chosen = np.random.default_rng(seed).permutation(len(pool))[:OUTSIDE_SAMPLE]
    close = 0
    for k in chosen:
        ctx, i = pool[int(k)]
        core = ctx.parts[i][1]
        own = ctx.counts[core.casefold()]
        close += any(d == 1 and ctx.counts[w] >= NEIGHBOUR_RATIO * max(own, 1) for w, d in ctx.near_vocabulary(core))
    return {"right_plain_outside_dictionary": len(pool), "sampled": len(chosen), "with_stronger_neighbour": int(close)}


def oracles(cases: list[PageCase]) -> dict:
    """What a join, a segmentation and a drop could reach, from the labels alone."""
    join: Counter = Counter()
    segment: Counter = Counter()
    drop: Counter = Counter()
    books = [dictionary.load(language) for language in lexicon.available(LANGUAGES)]
    for case in cases:
        index = {w.id: i for i, w in enumerate(case.words)}
        groups: dict[tuple, list[int]] = {}
        for word, label in zip(case.words, case.labels, strict=True):
            kind = error_kind(label, word)
            if kind == "split":
                groups.setdefault(label.truth_ids, []).append(index[label.ocr_id])
            elif kind == "merge":
                segment.update(_segment_reach(word.text, label.truth_text, books))
            elif kind == "invented":
                drop["tokens"] += 1
                drop["short"] += len(word.text) <= 2
        for ids, members in groups.items():
            members.sort()
            truth = normalise(next(w.text for w in case.truth if w.id == ids[0]))
            texts = [normalise(case.words[i].text) for i in members]
            joined = "".join(texts)
            line = {case.words[i].line_id for i in members}
            adjacent = members == list(range(members[0], members[0] + len(members)))
            join["groups"] += 1
            join["tokens"] += len(members)
            join["same_line_and_adjacent"] += len(line) == 1 and adjacent
            join["exact"] += joined == truth
            join["within_1_edit"] += Levenshtein.distance(joined, truth) <= 1
            join["letters"] += letters(joined) == letters(truth)
            join["two_or_three_tokens"] += 2 <= len(members) <= 3
    return {
        "join": {**join, "exact_share": share(join["exact"], join["groups"])},
        "segment": {
            **segment,
            "reachable_share": share(segment["reachable"], segment["tokens"]),
            "within_2_edits_of_no_space_share": share(segment["within_2_edits_of_no_space"], segment["tokens"]),
        },
        "drop": {**drop, "short_share": share(drop["short"], drop["tokens"])},
    }


def _segment_reach(text: str, truth_text: str | None, books) -> Counter:
    out = Counter(tokens=1)
    if not truth_text:
        return out
    pieces = truth_text.split(" ")
    out["truth_words"] += len(pieces)
    token = normalise(text)
    glued = "".join(pieces)
    out["only_miss_the_space"] += token == glued
    out["within_2_edits_of_no_space"] += Levenshtein.distance(token, glued) <= 2
    cores = [letters(p) for p in pieces]
    listed = all(
        (len(core) == 1 and core in SINGLE_LETTER_WORDS) or any(book.rank_of(core) >= 0 for book in books)
        for core in cores
    )
    plain = split_word(text) is not None and len(letters(text)) >= 5
    out["reachable"] += plain and listed and letters(text) == "".join(cores) and 2 <= len(pieces) <= 3
    return out


DEFINITIONS = {
    "wrong_one_to_one": "OCR word and truth word paired one to one, different, no digit on either side",
    "plain": "the token is letters with punctuation attached at its ends (v4's rule)",
    "sites": "plain tokens that no protection rule excludes (candidates.PageContext.is_site)",
    "written": "the truth, as written, is among the proposed texts",
    "letters": "some proposed text has the truth's letters (case, accents kept; punctuation ignored)",
    "dictionary_letters_ceiling": "the truth is a listed word within 3 edits of the token's letters: the figure that "
    "reproduces the 89-92 % of the design",
    "pool": "every source before deduplication and before the cap of 16",
}


def report(data_dir: Path, set_names) -> dict:
    out: dict = {"definitions": DEFINITIONS}
    for set_name in set_names:
        cases, counts = load_cases(data_dir, [set_name])
        sites = wrong_sites(cases)
        out[set_name] = {
            "counts": dict(counts),
            "errors": error_table(cases),
            "pool": pool_table(sites, cases),
            "pool_without_punctuation_context": pool_table(wrong_sites(cases, context_rule=False), cases),
            "page_vocabulary_harm": outside_dictionary_neighbours(cases),
            "oracles": oracles(cases),
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sets", nargs="+", default=["train", "val"])
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--out", type=Path, default=Path("bench/results/pool-recall-cached.json"))
    args = parser.parse_args(argv)
    result = report(args.data, args.sets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({name: result[name]["pool"] for name in args.sets}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
