from dataclasses import dataclass
from typing import Any


@dataclass
class Collection:
    id: int
    name: str

    @staticmethod
    def from_dict(data: dict) -> 'Collection':
        return Collection(
            id=data.get("id"),
            name=data.get('name', '')
        )