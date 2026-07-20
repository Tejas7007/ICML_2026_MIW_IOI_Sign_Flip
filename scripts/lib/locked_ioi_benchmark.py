"""Deterministic split-safe IOI benchmark used by appendix controls.

The benchmark contains 15 canonical IOI template families. Every sampled base
item produces matched ABBA and BABA prompts and three controls:

* ``dedup`` replaces only S2 with a third name;
* ``dedup_alt`` uses a different third name;
* ``placebo`` changes a neutral filler while preserving the repeated name.

The test, selection, validation, and development splits use disjoint name sets
and disjoint template families. The default test split has 800 examples and
reproduces prompt hash
``34d4fd78419110f21e70f8129a84d992cc6b10d02ddaa4c5d172c6d586ad0553``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import random
from typing import Literal

Order = Literal["ABBA", "BABA"]


@dataclass(frozen=True)
class TemplateFamily:
    idx: int
    abba: str
    baba: str
    placebo_kind: Literal["place", "arg_adj"]

    def template(self, order: Order) -> str:
        return self.abba if order == "ABBA" else self.baba


TEMPLATE_FAMILIES: tuple[TemplateFamily, ...] = (
    TemplateFamily(0, "Then, {IO} and {S1} went to the {PLACE}. {S2} gave a {OBJECT} to",
                   "Then, {S1} and {IO} went to the {PLACE}. {S2} gave a {OBJECT} to", "place"),
    TemplateFamily(1, "Then, {IO} and {S1} had a lot of fun at the {PLACE}. {S2} gave a {OBJECT} to",
                   "Then, {S1} and {IO} had a lot of fun at the {PLACE}. {S2} gave a {OBJECT} to", "place"),
    TemplateFamily(2, "Then, {IO} and {S1} were working at the {PLACE}. {S2} decided to give a {OBJECT} to",
                   "Then, {S1} and {IO} were working at the {PLACE}. {S2} decided to give a {OBJECT} to", "place"),
    TemplateFamily(3, "Then, {IO} and {S1} were thinking about going to the {PLACE}. {S2} wanted to give a {OBJECT} to",
                   "Then, {S1} and {IO} were thinking about going to the {PLACE}. {S2} wanted to give a {OBJECT} to", "place"),
    TemplateFamily(4, "Then, {IO} and {S1} had a {ARG_ADJ} argument, and afterwards {S2} said to",
                   "Then, {S1} and {IO} had a {ARG_ADJ} argument, and afterwards {S2} said to", "arg_adj"),
    TemplateFamily(5, "After {IO} and {S1} went to the {PLACE}, {S2} gave a {OBJECT} to",
                   "After {S1} and {IO} went to the {PLACE}, {S2} gave a {OBJECT} to", "place"),
    TemplateFamily(6, "When {IO} and {S1} got a {OBJECT} at the {PLACE}, {S2} decided to give it to",
                   "When {S1} and {IO} got a {OBJECT} at the {PLACE}, {S2} decided to give it to", "place"),
    TemplateFamily(7, "When {IO} and {S1} got a {OBJECT} at the {PLACE}, {S2} decided to give the {OBJECT} to",
                   "When {S1} and {IO} got a {OBJECT} at the {PLACE}, {S2} decided to give the {OBJECT} to", "place"),
    TemplateFamily(8, "While {IO} and {S1} were working at the {PLACE}, {S2} gave a {OBJECT} to",
                   "While {S1} and {IO} were working at the {PLACE}, {S2} gave a {OBJECT} to", "place"),
    TemplateFamily(9, "While {IO} and {S1} were commuting to the {PLACE}, {S2} gave a {OBJECT} to",
                   "While {S1} and {IO} were commuting to the {PLACE}, {S2} gave a {OBJECT} to", "place"),
    TemplateFamily(10, "After the lunch, {IO} and {S1} went to the {PLACE}. {S2} gave a {OBJECT} to",
                    "After the lunch, {S1} and {IO} went to the {PLACE}. {S2} gave a {OBJECT} to", "place"),
    TemplateFamily(11, "Afterwards, {IO} and {S1} went to the {PLACE}. {S2} gave a {OBJECT} to",
                    "Afterwards, {S1} and {IO} went to the {PLACE}. {S2} gave a {OBJECT} to", "place"),
    TemplateFamily(12, "Then, {IO} and {S1} had a {ARG_ADJ} argument. Afterwards {S2} said to",
                    "Then, {S1} and {IO} had a {ARG_ADJ} argument. Afterwards {S2} said to", "arg_adj"),
    TemplateFamily(13, "The {PLACE} {IO} and {S1} went to had a {OBJECT}. {S2} gave it to",
                    "The {PLACE} {S1} and {IO} went to had a {OBJECT}. {S2} gave it to", "place"),
    TemplateFamily(14, "Friends {IO} and {S1} found a {OBJECT} at the {PLACE}. {S2} gave it to",
                    "Friends {S1} and {IO} found a {OBJECT} at the {PLACE}. {S2} gave it to", "place"),
)

NAME_POOL: tuple[str, ...] = (
    "Aaron","Adam","Alan","Alex","Alice","Amanda","Amy","Andrew","Angela","Anna",
    "Anne","Arthur","Ben","Beth","Bill","Bob","Brad","Brian","Carl","Carol",
    "Charlie","Chris","Claire","Colin","Dan","Daniel","Dave","David","Dean","Diana",
    "Don","Donna","Ed","Edward","Elena","Ellen","Emily","Emma","Eric","Eva",
    "Frank","Fred","Gary","George","Glen","Grace","Greg","Hannah","Harry","Helen",
    "Henry","Holly","Ian","Jack","Jake","James","Jane","Jason","Jean","Jeff",
    "Jennifer","Jerry","Jim","Joan","Joe","John","Jon","Julia","Julie","Justin",
    "Karen","Kate","Keith","Kelly","Ken","Kevin","Kim","Larry","Laura","Lee","Leon",
    "Linda","Lisa","Louis","Lucy","Luke","Lynn","Marc","Maria","Marie","Mark",
    "Martin","Mary","Matt","Max","Meg","Michael","Mike","Nancy","Neil","Nick",
    "Noah","Oliver","Owen","Pat","Paul","Peter","Phil","Rachel","Ray","Richard",
    "Rob","Robert","Robin","Roger","Ron","Rose","Roy","Ruth","Ryan","Sam","Sarah",
    "Scott","Sean","Sharon","Simon","Sophie","Steve","Susan","Tim","Tom","Tony",
    "Victor","Will",
)
PLACES = ("store","park","school","office","garden","church","lake","beach")
PLACEBO_PLACES = ("yard","gate","room","shop","bank","hall","field","road")
OBJECTS = ("ring","ball","book","box","card","coin","cup","pen","hat","key","lamp",
           "phone","watch","bag","rock","drink")


def _hash_id(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class BaseItem:
    base_item_id: str
    split: str
    template_idx: int
    IO: str
    S: str
    third: str
    third_alt: str
    place: str
    placebo_place: str
    object_: str
    arg_adj: str = "long"
    placebo_arg_adj: str = "big"

    @property
    def pair_id(self) -> str:
        return _hash_id(*sorted((self.IO, self.S)))


@dataclass(frozen=True)
class IOIExample:
    base: BaseItem
    order: Order

    @property
    def example_id(self) -> str:
        return _hash_id(self.base.base_item_id, self.order)

    @property
    def base_item_id(self) -> str:
        return self.base.base_item_id

    @property
    def template_idx(self) -> int:
        return self.base.template_idx

    @property
    def pair_id(self) -> str:
        return self.base.pair_id

    def _render(self, *, s2: str, placebo: bool = False) -> str:
        family = TEMPLATE_FAMILIES[self.base.template_idx]
        place = self.base.placebo_place if placebo and family.placebo_kind == "place" else self.base.place
        arg_adj = self.base.placebo_arg_adj if placebo and family.placebo_kind == "arg_adj" else self.base.arg_adj
        return family.template(self.order).format(
            IO=self.base.IO, S1=self.base.S, S2=s2, PLACE=place,
            OBJECT=self.base.object_, ARG_ADJ=arg_adj,
        )

    def clean_prompt(self) -> str:
        return self._render(s2=self.base.S)

    def dedup_prompt(self) -> str:
        return self._render(s2=self.base.third)

    def dedup_alt_prompt(self) -> str:
        return self._render(s2=self.base.third_alt)

    def placebo_prompt(self) -> str:
        return self._render(s2=self.base.S, placebo=True)

    def to_dict(self) -> dict:
        return {
            **asdict(self.base),
            "order": self.order,
            "example_id": self.example_id,
            "pair_id": self.pair_id,
            "clean": self.clean_prompt(),
            "dedup": self.dedup_prompt(),
            "dedup_alt": self.dedup_alt_prompt(),
            "placebo": self.placebo_prompt(),
        }


@dataclass
class BenchmarkSplit:
    name: str
    names: list[str]
    template_indices: list[int]
    examples: list[IOIExample]

    def prompt_hash(self) -> str:
        h = hashlib.sha256()
        for ex in self.examples:
            h.update(ex.example_id.encode())
            h.update(b"\x00")
            for text in (
                ex.clean_prompt(), ex.dedup_prompt(),
                ex.dedup_alt_prompt(), ex.placebo_prompt(),
            ):
                h.update(text.encode())
                h.update(b"\x00")
        return h.hexdigest()


class IOIBenchmark:
    def __init__(
        self,
        seed: int = 20240601,
        n_test: int = 800,
        n_selection: int = 300,
        n_validation: int = 300,
        n_dev: int = 40,
    ):
        counts = {
            "test": n_test, "selection": n_selection,
            "validation": n_validation, "dev": n_dev,
        }
        if any(n <= 0 or n % 2 for n in counts.values()):
            raise ValueError("Every split size must be positive and even")
        self.seed = seed
        self.counts = counts
        self.rng = random.Random(seed)
        self._partition()
        self._build_all()

    def _partition(self) -> None:
        names = list(NAME_POOL)
        self.rng.shuffle(names)
        n = len(names)
        b1, b2, b3 = int(0.40*n), int(0.62*n), int(0.84*n)
        self.split_names = {
            "test": names[:b1], "selection": names[b1:b2],
            "validation": names[b2:b3], "dev": names[b3:],
        }
        tids = list(range(len(TEMPLATE_FAMILIES)))
        self.rng.shuffle(tids)
        self.split_templates = {
            "test": tids[:6], "selection": tids[6:9],
            "validation": tids[9:12], "dev": tids[12:15],
        }

    def _make_base(self, split: str) -> BaseItem:
        io_name, s_name, third, third_alt = self.rng.sample(self.split_names[split], 4)
        tidx = self.rng.choice(self.split_templates[split])
        place = self.rng.choice(PLACES)
        placebo_place = self.rng.choice(PLACEBO_PLACES)
        obj = self.rng.choice(OBJECTS)
        base_id = _hash_id(split, tidx, io_name, s_name, third, third_alt, place, placebo_place, obj)
        return BaseItem(base_id, split, tidx, io_name, s_name, third, third_alt,
                        place, placebo_place, obj)

    def _build_split(self, split: str) -> BenchmarkSplit:
        target_pairs = self.counts[split] // 2
        bases: list[BaseItem] = []
        seen: set[str] = set()
        while len(bases) < target_pairs:
            base = self._make_base(split)
            if base.base_item_id not in seen:
                seen.add(base.base_item_id)
                bases.append(base)
        examples = [
            IOIExample(base, order)
            for base in bases for order in ("ABBA", "BABA")
        ]
        self.rng.shuffle(examples)
        return BenchmarkSplit(
            split, self.split_names[split],
            self.split_templates[split], examples,
        )

    def _build_all(self) -> None:
        self.splits = {
            split: self._build_split(split)
            for split in ("test", "selection", "validation", "dev")
        }

    def split(self, name: str) -> BenchmarkSplit:
        return self.splits[name]
