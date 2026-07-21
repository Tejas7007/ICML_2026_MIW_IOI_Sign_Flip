"""Deterministic IOI prompts for the matched-S2 intervention.

The paper's core intervention uses the first ten BABA-style template families
from the project dataset, with 30 prompts per family. Each sampled name pair is
emitted in both role assignments. Prompt sampling intentionally mirrors the
original producer, including the otherwise redundant one-item template draw,
so seed 42 reproduces the same prompt sequence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    """One IOI prompt and its matched non-repeating S2 control."""

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
    return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]


def single_token_words(tokenizer: Any, words: Iterable[str]) -> list[str]:
    """Return words represented by one token when preceded by a space."""

    return [word for word in words if len(_token_ids(tokenizer, " " + word)) == 1]


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
    """Materialize the exact ten-template, 300-prompt intervention set."""

    if prompts_per_template <= 0 or prompts_per_template % 2:
        raise ValueError("prompts_per_template must be a positive even integer")

    names = single_token_words(tokenizer, CANDIDATE_NAMES)
    places = single_token_words(tokenizer, PLACES)
    objects = single_token_words(tokenizer, OBJECTS)
    if len(names) < 3:
        raise RuntimeError("Fewer than three candidate names are single-token")
    if not places or not objects:
        raise RuntimeError("No single-token place or object survived tokenization")

    records: list[PromptRecord] = []
    donor_rng = np.random.default_rng(control_seed)

    for template_id, template in enumerate(BABA_TEMPLATES):
        rng = random.Random(seed)
        for pair_id in range(prompts_per_template // 2):
            # The original IOIDataset calls rng.choice([template]) before every
            # sampled pair. Although the result is fixed, consuming this draw is
            # necessary for byte-for-byte prompt-sequence compatibility.
            selected_template = rng.choice((template,))
            first, second = rng.sample(names, 2)
            place = rng.choice(places)
            obj = rng.choice(objects)

            for orientation, io_name, s_name in (
                ("A_io_B_s", first, second),
                ("B_io_A_s", second, first),
            ):
                prompt = _render(selected_template, io_name, s_name, place, obj)
                eligible = [name for name in names if name not in (io_name, s_name)]
                control_name = str(donor_rng.choice(np.asarray(eligible, dtype=object)))
                second_index = prompt.rfind(s_name)
                if second_index < 0:
                    raise RuntimeError("Could not locate the second repeated-name span")
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

    expected = len(BABA_TEMPLATES) * prompts_per_template
    if len(records) != expected:
        raise AssertionError(f"Expected {expected} prompts, produced {len(records)}")
    return records


def prompt_manifest_hash(records: Sequence[PromptRecord]) -> str:
    """Return a stable SHA-256 over the aligned prompt metadata."""

    payload = [asdict(record) for record in records]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def locate_s2_positions(tokenizer: Any, records: Sequence[PromptRecord]) -> list[int]:
    """Locate S2 in token coordinates before a beginning-of-sequence token."""

    positions: list[int] = []
    for record in records:
        tokens = _token_ids(tokenizer, record.prompt)
        s_tokens = _token_ids(tokenizer, " " + record.s_name)
        if len(s_tokens) != 1:
            raise RuntimeError(f"{record.s_name!r} is not one token in context")
        matches = [index for index, token in enumerate(tokens) if token == s_tokens[0]]
        if len(matches) != 2:
            raise RuntimeError(
                f"{record.example_id} has {len(matches)} repeated-name token occurrences"
            )
        positions.append(matches[1])
    return positions
