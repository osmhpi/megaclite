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
import ipywidgets
from IPython.core.display import HTML, display, clear_output
from IPython.core.magic import (
    Magics,
    cell_magic,
    line_magic,
    magics_class,
    needs_local_scope,
)

from .messages import (
    AbbortJob,
    BashJob,
    ClientInfo,
    JobResult,
    TrainingJob,
    JobInfo,
    JobState,
    StdOut,
)
from pip._internal.operations import freeze
import logging

logging.basicConfig(format="%(asctime)s %(message)s")


def collect_client_info() -> ClientInfo:
    return ClientInfo(
        python_version=sys.version.split(" ")[0], packages=list(freeze.freeze())
    )

COMPUTE_CONFIGS = ["1","2","3","4","7"]

@magics_class
class RemoteTrainingMagics(Magics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host: str = "127.0.0.1"
        self.port: str = 6001
        self.key: str = None
        self.message_box = None

    def print(self, value: str):
        self.message_box.value = value

    def init_print(self):
        clear_output()
        self.message_box = ipywidgets.HTML()
        display(self.message_box)

    def send_job(self, job, on_success: Optional[callable] = None):
        address = (self.host, self.port)
        conn = Client(address)
        try:
            logging.info("sending job start")
            conn.send(job)
            logging.info("sending job finished")
            job_uuid = None
            
            while message := conn.recv():
                if isinstance(message, JobInfo):
                    job_uuid = message.uuid
                    if message.state == JobState.PENDING:
                        self.print(
                            f"<i>You are number {message.no_in_queue + 1} in the queue.<i>"
                        )
                    elif message.state == JobState.STARTED:
                        logging.info("processing job started")
                        self.print(f"<i>Job started executing.<i>")
                    elif message.state == JobState.FINISHED:
                        logging.info("processing job finished")
                        self.print(f"<i>Retrieving weights from remote.<i>")
                        if on_success:
                            logging.info("retrieving weights from remote")
                            result = conn.recv()
                            logging.info("retrieving weights from remote finished")
                            on_success(result)
                    if message.state.exited:
                        self.print(f"<i>Job exited with status {str(message.state)}.<i>")
                        break
                elif isinstance(message, StdOut):
                    print(message.line, end="")
        except KeyboardInterrupt:
            self.print(f"<i>Aborting job with uuid {job_uuid}.<i>")
            conn.send(AbortJob(uuid=job_uuid))
            result = conn.recv()
            if result.state == JobState.ABORTED:
                self.print(f"<i>Job with uuid {job_uuid} was aborted.<i>")

    def send_training_job(self, cell: str, model, model_name: str, mig_slices: int):
        logging.info("new training job")
        file = BytesIO()
        dill.dump_module(file)
        self.print(f"<i>Sending {len(file.getbuffer())/2**20:.0f}MB of state.<i>")
        job = TrainingJob(
            cell, model_name, file.getvalue(), client=collect_client_info(),mig_slices=mig_slices
        )

        def success_handler(result: JobResult):
            weights_file = BytesIO(result.result)
            device = torch.device("cpu")
            model.load_state_dict(torch.load(weights_file, map_location=device))
            logging.info("loading weights finished")

        self.send_job(job=job, on_success=success_handler)

    # @line_magic
    # def remote_config(self, line):
    #     """Use: %remote_config <host> <port> <key>"""
    #     parser = argparse.ArgumentParser(description="Process some integers.")
    #     parser.add_argument("host", type=str)
    #     parser.add_argument("port", type=int)
    #     # parser.add_argument("key", type=str)

    #     args = parser.parse_args(shlex.split(line))

    #     self.host = args.host
    #     self.port = args.port
    #     # self.key = args.key

    @line_magic
    def run_remote(self, line):
        """Use: %remote_config <host> <port> <key>"""
        self.init_print()
        self.print(f"<i>executing command `{line}` on remote host</i>")
        job = BashJob(command=line, client=collect_client_info())
        print(job.client.packages)
        self.send_job(job=job)

    @cell_magic
    @needs_local_scope
    def train_remote(self, line, cell, local_ns):
        """Use: %%remote <model> [<compute-slices>]"""
        self.init_print()
        
        parser = argparse.ArgumentParser(description="Remote training job args.")
        parser.add_argument("model", type=str)
        # parser.add_argument('-m', '--mig',
        #             action='store_true')
        # parser.add_argument("memory", choices=MEMORY_CONFIGS)
        parser.add_argument("compute", choices=COMPUTE_CONFIGS, nargs="?")

        args = parser.parse_args(shlex.split(line))

        model_name = args.model
        shared_text = "" # "shared" if args.shared else "dedicated"
        self.print(f"<i>training <b>{model_name}</b> on a <b>{shared_text}</b> remote gpu</i>"+ \
                   f"<br><i>MIG: requesting <b>{args.compute}</b> compute slices<i>"
                   )
        # <b>{args.memory}</b> of memory and 
        self.send_training_job(cell, local_ns[model_name], model_name, int(args.compute) if args.compute else None)

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
