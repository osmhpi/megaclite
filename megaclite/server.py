"""This module implements the server for the remote GPU extension."""
import os
import subprocess
import tempfile
from datetime import datetime
from multiprocessing import Process, Queue
from multiprocessing.connection import Connection, Listener
from pathlib import Path

import click

from .messages import BashJob, JobResult, TrainingJob, JobInfo, JobState, StdOut


EXCLUDED_PACKAGES = ["megaclite"]
ADDITIONAL_PACKAGES = ["click"]


def install_python_version(version: str):
    """Install the requested python version."""
    # shell injection waiting to happen :)
    subprocess.run(["pyenv", "install", version, "-s"], check=True)


def get_tmp_dir(sub_dir=None):
    """Create a new temporary directory."""
    if sub_dir is None:
        sub_dir = datetime.now().isoformat()
    tmp_path = Path(tempfile.gettempdir(), "megaclite", sub_dir)
    tmp_path.mkdir(exist_ok=True, parents=True)
    return tmp_path


def get_venv(tmp_dir):
    """Return path to venv."""
    return tmp_dir / "venv"


def get_pip(tmp_dir):
    """Return path to pip."""
    return get_venv(tmp_dir) / "bin/pip"


def get_python(tmp_dir):
    """Return path to python interpreter."""
    return get_venv(tmp_dir) / "bin/python"


def get_state_file(tmp_dir):
    """Return path to state file."""
    return tmp_dir / "state.pkl"


def get_cell_file(tmp_dir):
    """Return path to cell file."""
    return tmp_dir / "cell.py"


def get_output_file(tmp_dir):
    """Return path to output."""
    return tmp_dir / "output.pt"


def get_python_with_version(version):
    return Path.home() / f".pyenv/versions/{version}/bin/python3"


import hashlib


def create_venv_with_requirements(version, requirements: list[str]):
    """Create a new venv with the requested python version and packages."""
    print("creating venv with python version", version)

    requirements = [
        r for r in requirements if r.split("==")[0] not in EXCLUDED_PACKAGES
    ]
    requirements.extend(ADDITIONAL_PACKAGES)
    message = hashlib.sha256()
    message.update(version.encode())
    for req in sorted(requirements):
        message.update(req.encode())

    tmp_path = get_tmp_dir(message.hexdigest())
    if get_venv(tmp_path).exists():
        return tmp_path

    print("creating venv in", str(get_venv(tmp_path)))
    subprocess.run(
        [get_python_with_version(version), "-m", "venv", str(get_venv(tmp_path))],
        check=True,
    )
    print("installing packages")
    subprocess.run(
        [str(get_pip(tmp_path)), "install", "-r", "/dev/stdin"],
        input="\n".join(requirements),
        text=True,
        check=True,
    )
    subprocess.run(
        [str(get_pip(tmp_path)), "install", "."],
        text=True,
        check=True,
    )
    return tmp_path


# pylint: disable=too-few-public-methods
class RemoteStdout:
    """Wraps a multiprocessing.connection.Connection to be file compatible."""

    def __init__(self, conn: Connection) -> None:
        self.conn = conn

    def write(self, text):
        """Send the text back to the client."""
        self.conn.send(StdOut(text))


# def reexecute_and_train(state_file: str, cell: str, model_name: str, queue):
#     """This function runs in the subprocess."""
#     # import these modules only in the subprocess
#     import dill  # pylint: disable=import-outside-toplevel
#     import torch  # pylint: disable=import-outside-toplevel

#     # load state from the users notebook
#     file = BytesIO(state_file)
#     dill.load_module(file)

#     # train the model
#     exec(cell)  # pylint: disable=exec-used

#     # save the model
#     out_file = BytesIO()
#     torch.save(globals()[model_name].state_dict(), out_file)
#     queue.put(out_file.getvalue())


def execute_in_subprocess(tmp_dir: Path, job: TrainingJob, conn: Connection):
    """Setup the subprocess execution with stdout redirect."""

    state_file = get_state_file(tmp_dir)
    cell_file = get_cell_file(tmp_dir)
    output_file = get_output_file(tmp_dir)

    state_file.write_bytes(job.state)
    cell_file.write_text(job.cell)
    print(get_python(tmp_dir))
    with subprocess.Popen(
        [
            get_python(tmp_dir),
            "-m",
            "megaclite._runtime",
            str(state_file),
            str(cell_file),
            job.model_name,
            str(output_file),
        ],
        stdout=subprocess.PIPE,
        text=True,
        cwd=str(tmp_dir),
    ) as process:
        conn.send(JobInfo(state=JobState.STARTED, no_in_queue=0))

        for line in iter(process.stdout.readline, ""):
            conn.send(StdOut(line))

    conn.send(JobInfo(state=JobState.FINISHED, no_in_queue=0))
    conn.send(JobResult(result=output_file.read_bytes()))


def execute_bash_script(tmp_dir: Path, job: BashJob, conn: Connection):
    with subprocess.Popen(
        ["/bin/bash", "-c", job.command],
        stdout=subprocess.PIPE,
        text=True,
        cwd=str(tmp_dir),
    ) as process:
        conn.send(JobInfo(state=JobState.STARTED, no_in_queue=0))

        for line in iter(process.stdout.readline, ""):
            conn.send(StdOut(line))

    conn.send(JobInfo(state=JobState.FINISHED, no_in_queue=0))


def worker_main(queue):
    """The main worker thread."""
    while True:
        message, conn = queue.get()
        install_python_version(message.client.python_version)
        tmp_dir = create_venv_with_requirements(
            message.client.python_version, message.client.packages
        )
        execute_in_subprocess(tmp_dir, message, conn)


@click.command()
@click.option("-h", "--host", default="127.0.0.1")
@click.option("-p", "--port", default=6001, type=int)
# @click.option("--password", prompt=True, hide_input=True)
def main(host, port):
    """The main function"""
    listener = Listener((host, port))

    jobs = Queue()
    worker = Process(target=worker_main, args=(jobs,))
    worker.start()

    while True:
        try:
            conn = listener.accept()
            message = conn.recv()
            if isinstance(message, TrainingJob):
                print(
                    "got new TrainingJob",
                    listener.last_accepted,
                    f"#{jobs.qsize()} in queue",
                )
                conn.send(JobInfo(state=JobState.PENDING, no_in_queue=jobs.qsize()))
            elif isinstance(message, BashJob):
                print(
                    "got new BashJob",
                    listener.last_accepted,
                    f"#{jobs.qsize()} in queue",
                )
                conn.send(JobInfo(state=JobState.PENDING, no_in_queue=jobs.qsize()))
            jobs.put((message, conn))
        except KeyboardInterrupt:
            print("got Ctrl+C, cleaning up")
            listener.close()
            worker.join()
            break
        except Exception:  # pylint: disable=broad-exception-caught
            listener.close()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
