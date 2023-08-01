from dataclasses import dataclass, field
import enum
from typing import Optional


@dataclass
class ClientInfo:
    python_version: str
    packages: Optional[list[str]] = field(default_factory=list)


@dataclass
class TrainingJob:
    cell: str
    model_name: str
    state: bytes
    client: ClientInfo

class BashJob:
    command: str


@dataclass
class StdOut:
    line: str


class EOF:
    pass


class JobState(enum.Enum):
    PENDING = 1
    STARTED = 2
    REJECTED = 3
    FINISHED = 4
    ABBORTED = 5

    @property
    def exited(self):
        return self in set([JobState.REJECTED, JobState.FINISHED, JobState.ABBORTED])


@dataclass
class JobInfo:
    state: JobState
    no_in_queue: int
    result: Optional[bytes] = None
