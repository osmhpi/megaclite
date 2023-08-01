import sys
import argparse
import shlex
from datetime import datetime
from io import BytesIO
from multiprocessing.connection import Client
from pathlib import Path
from typing import Optional

import dill
import torch
from IPython.core.display import HTML, display
from IPython.core.magic import (
    Magics,
    cell_magic,
    line_magic,
    magics_class,
    needs_local_scope,
)

from .messages import BashJob, ClientInfo, TrainingJob, JobInfo, JobState, StdOut
from pip._internal.operations import freeze


def collect_client_info() -> ClientInfo:
    return ClientInfo(
        python_version=sys.version.split(" ")[0], packages=list(freeze.freeze())
    )


@magics_class
class RemoteTrainingMagics(Magics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host: str = "127.0.0.1"
        self.port: str = 6001
        self.key: str = None
    
    def send_job(self, job, on_success:Optional[callable] = None):
        address = (self.host, self.port)
        conn = Client(address)
        conn.send(job)

        while message := conn.recv():
            if isinstance(message, JobInfo):
                if message.state == JobState.PENDING:
                    display(
                        HTML(
                            f"<i>You are number {message.no_in_queue + 1} in the queue.<i>"
                        )
                    )
                elif message.state == JobState.STARTED:
                    display(HTML(f"<i>Job started executing.<i>"))
                elif message.state == JobState.FINISHED:
                    display(HTML(f"<i>Retrieving weights from remote.<i>"))
                    if on_success:
                        on_success(message)
                if message.state.exited:
                    display(HTML(f"<i>Job exited with status {str(message.state)}.<i>"))
                    break
            elif isinstance(message, StdOut):
                print(message.line, end="")

    def send_training_job(self, cell: str, model, model_name: str):
        file = BytesIO()
        dill.dump_module(file)
        display(HTML(f"<i>Sending {len(file.getbuffer())/2**20:.0f}MB of state.<i>"))
        job = TrainingJob(cell, model_name, file.getvalue(), client=collect_client_info())

        def success_handler(message):
            weights_file = BytesIO(message.result)
            device = torch.device("cpu")
            model.load_state_dict(torch.load(weights_file, map_location=device))
        
        self.send_job(job=job, on_success=success_handler)
       

    @line_magic
    def remote_config(self, line):
        """Use: %remote_config <host> <port> <key>"""
        parser = argparse.ArgumentParser(description="Process some integers.")
        parser.add_argument("host", type=str)
        parser.add_argument("port", type=int)
        # parser.add_argument("key", type=str)

        args = parser.parse_args(shlex.split(line))

        self.host = args.host
        self.port = args.port
        # self.key = args.key

    @line_magic
    def run_remote(self, line):
        """Use: %remote_config <host> <port> <key>"""
        display(HTML(f"<i>executing command `{line}` on remote host</i>"))
        job = BashJob(command=line, client=collect_client_info())
        self.send_job(job=job)

    @cell_magic
    @needs_local_scope
    def train_remote(self, line, cell, local_ns):
        """Use: %%remote <model>"""
        if not line:
            raise TypeError("%%remote missing 1 required positional argument: 'model'")

        model_name = line.strip()
        display(HTML(f"<i>training `{model_name}` on remote gpu</i>"))
        self.send_training_job(cell, local_ns[model_name], model_name)

    # @line_cell_magic
    # def lcmagic(self, line, cell=None):
    #     "Magic that works both as %lcmagic and as %%lcmagic"
    #     if cell is None:
    #         print("Called as line magic")
    #         return line
    #     else:
    #         print("Called as cell magic")
    #         return line, cell


def pre_run_hook(info):
    info.raw_cell


def load_ipython_extension(ipython):
    ipython.register_magics(RemoteTrainingMagics)
    # ipython.events.register("pre_run_cell", pre_run_hook)
