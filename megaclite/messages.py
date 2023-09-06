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
    mig_slices: int
    uuid: Optional[str] = None

@dataclass
class AbortJob:
    uuid: str


@dataclass
class ShellJob:
    command: str
    client: ClientInfo
    uuid: Optional[str] = None


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
    ABORTED = 5

    @property
    def exited(self):
        return self in set([JobState.REJECTED, JobState.FINISHED, JobState.ABORTED])


@dataclass
class JobInfo:
    state: JobState
    no_in_queue: int
    uuid: str


@dataclass
class JobResult:
    result: bytes
