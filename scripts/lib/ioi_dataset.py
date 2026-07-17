"""Deterministic IOI prompt materialization for the camera-ready protocol.

The historical cross-scale experiments evaluated the first ten BABA-style
template families from the project dataset module. Each family contributes
30 prompts generated with seed 42. Prompts are emitted in symmetric pairs:
for a sampled pair of names, the two identities exchange the IO and repeated
subject roles while the sentence template remains fixed.

This module intentionally documents that exact protocol. It does not describe
the prompt set as ABBA/BABA balanced.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Iterable, Sequence

import numpy as np

BABA_TEMPLATES: tuple[str, ...] = (
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to",
)

CANDIDATE_NAMES: tuple[str, ...] = (
    "Aaron", "Adam", "Alan", "Alex", "Alice", "Amanda", "Amy", "Andrew",
    "Angela", "Anna", "Anne", "Arthur", "Ben", "Beth", "Bill", "Bob",
    "Brad", "Brian", "Carl", "Carol", "Charlie", "Chris", "Claire", "Colin",
    "Dan", "Daniel", "Dave", "David", "Dean", "Diana", "Don", "Donna",
    "Ed", "Edward", "Elena", "Ellen", "Emily", "Emma", "Eric", "Eva",
    "Frank", "Fred", "Gary", "George", "Glen", "Grace", "Greg", "Hannah",
    "Harry", "Helen", "Henry", "Holly", "Ian", "Iris", "Jack", "Jake",
    "James", "Jane", "Jason", "Jean", "Jeff", "Jennifer", "Jerry", "Jim",
    "Joan", "Joe", "John", "Jon", "Julia", "Julie", "Justin", "Karen",
    "Kate", "Keith", "Kelly", "Ken", "Kevin", "Kim", "Larry", "Laura",
    "Lee", "Leon", "Linda", "Lisa", "Louis", "Lucy", "Luke", "Lynn",
    "Marc", "Maria", "Marie", "Mark", "Martin", "Mary", "Matt", "Max",
    "Meg", "Michael", "Mike", "Nancy", "Neil", "Nick", "Noah", "Oliver",
    "Owen", "Pat", "Paul", "Peter", "Phil", "Rachel", "Ray", "Richard",
    "Rob", "Robert", "Robin", "Roger", "Ron", "Rose", "Roy", "Ruth",
    "Ryan", "Sam", "Sarah", "Scott", "Sean", "Sharon", "Simon", "Sophie",
    "Steve", "Susan", "Tim", "Tom", "Tony", "Victor", "Will", "Zoe",
)

PLACES: tuple[str, ...] = (
    "store", "market", "garden", "museum", "library", "school", "church",
    "park", "beach", "lake", "forest", "river", "mountain", "castle",
    "theater", "restaurant", "hospital", "airport", "zoo", "gym",
)

OBJECTS: tuple[str, ...] = (
    "ring", "ball", "book", "bottle", "box", "card", "coin", "cup",
    "flower", "gift", "hat", "key", "lamp", "letter", "pen", "phone",
    "photo", "shirt", "shoe", "watch",
)


@dataclass(frozen=True)
class PromptRecord:
    """One materialized original prompt and its matched de-duplicated control."""

    example_id: str
    template_id: int
    pair_id: int
    orientation: str
    prompt: str
    control_prompt: str
    io_name: str
    s_name: str
    control_name: str


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, list):
        ids = list(ids)
    return [int(x) for x in ids]


def single_token_words(tokenizer: Any, words: Iterable[str]) -> list[str]:
    """Keep words that occupy one token when preceded by a space."""

    kept: list[str] = []
    for word in words:
        if len(_token_ids(tokenizer, " " + word)) == 1:
            kept.append(word)
    return kept


def _render(template: str, io_name: str, s_name: str, place: str, obj: str) -> str:
    return (
        template.replace("[A]", io_name)
        .replace("[B]", s_name)
        .replace("[PLACE]", place)
        .replace("[OBJECT]", obj)
    )


def build_prompt_records(
    tokenizer: Any,
    *,
    prompts_per_template: int = 30,
    seed: int = 42,
    control_seed: int = 43,
) -> list[PromptRecord]:
    """Materialize the historical 10-template, 300-prompt protocol exactly."""

    if prompts_per_template <= 0 or prompts_per_template % 2:
        raise ValueError("prompts_per_template must be a positive even integer")

    names = single_token_words(tokenizer, CANDIDATE_NAMES)
    places = single_token_words(tokenizer, PLACES)
    objects = single_token_words(tokenizer, OBJECTS)
    if len(names) < 3:
        raise RuntimeError("Fewer than three single-token names survived tokenization")
    if not places or not objects:
        raise RuntimeError("No single-token place or object survived tokenization")

    records: list[PromptRecord] = []
    donor_rng = np.random.default_rng(control_seed)

    for template_id, template in enumerate(BABA_TEMPLATES):
        rng = random.Random(seed)
        pair_count = prompts_per_template // 2
        for pair_id in range(pair_count):
            first, second = rng.sample(names, 2)
            place = rng.choice(places)
            obj = rng.choice(objects)

            for orientation, io_name, s_name in (
                ("A_io_B_s", first, second),
                ("B_io_A_s", second, first),
            ):
                prompt = _render(template, io_name, s_name, place, obj)
                eligible = [n for n in names if n not in (io_name, s_name)]
                # The historical producer used one NumPy generator across all
                # template batches, with one donor draw per prompt.
                control_name = str(donor_rng.choice(np.asarray(eligible, dtype=object)))
                second_index = prompt.rfind(s_name)
                if second_index < 0:
                    raise RuntimeError("Could not locate S2 in prompt")
                control_prompt = (
                    prompt[:second_index]
                    + control_name
                    + prompt[second_index + len(s_name):]
                )

                records.append(
                    PromptRecord(
                        example_id=f"t{template_id:02d}-p{pair_id:02d}-{orientation}",
                        template_id=template_id,
                        pair_id=pair_id,
                        orientation=orientation,
                        prompt=prompt,
                        control_prompt=control_prompt,
                        io_name=io_name,
                        s_name=s_name,
                        control_name=control_name,
                    )
                )

    if len(records) != len(BABA_TEMPLATES) * prompts_per_template:
        raise AssertionError("Unexpected prompt count")
    return records


def prompt_manifest_hash(records: Sequence[PromptRecord]) -> str:
    """Return a stable SHA-256 over the paper-relevant prompt metadata."""

    payload = [
        {
            "example_id": r.example_id,
            "template_id": r.template_id,
            "pair_id": r.pair_id,
            "orientation": r.orientation,
            "prompt": r.prompt,
            "control_prompt": r.control_prompt,
            "io_name": r.io_name,
            "s_name": r.s_name,
            "control_name": r.control_name,
        }
        for r in records
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def locate_s2_positions(tokenizer: Any, records: Sequence[PromptRecord]) -> list[int]:
    """Locate the second occurrence of the repeated-name token in each prompt."""

    positions: list[int] = []
    for record in records:
        tokens = _token_ids(tokenizer, record.prompt)
        s_tokens = _token_ids(tokenizer, " " + record.s_name)
        if len(s_tokens) != 1:
            raise RuntimeError(f"{record.s_name!r} is not a single token in context")
        s_id = s_tokens[0]
        matches = [i for i, token_id in enumerate(tokens) if token_id == s_id]
        if len(matches) != 2:
            raise RuntimeError(
                f"{record.example_id} has {len(matches)} occurrences of the S token"
            )
        positions.append(matches[1])
    return positions
