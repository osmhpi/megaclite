from multiprocessing.connection import Listener
from multiprocessing import Process, Queue, set_start_method
from datetime import datetime
from pathlib import Path


def get_output_file(state_file: str)->str:
    return state_file + "-out"

from io import BytesIO


def reexecute_and_train(state_file: str, cell: str, model_name: str, queue):
    import dill
    import torch
    # output_file = get_output_file(state_file)
    # load state from the users notebook
    file = BytesIO(state_file)
    dill.load_module(file)
    # train the model
    exec(cell)
    # save the model
    out_file = BytesIO()
    torch.save(globals()[model_name].state_dict(), out_file)
    queue.put(out_file.getvalue())

def execute_in_subprocess(state_file, cell, model_name, stdout_listener):
    from contextlib import redirect_stdout
    # set_start_method("spawn")
    queue = Queue()
    p = Process(target=reexecute_and_train, args=(state_file, cell, model_name, queue))
    import sys
    class DummyStdout:
        def __init__(self) -> None:
            self.conn = stdout_listener.accept()
        def write(self, text):
            self.conn.send(text)
    dummy_stout = DummyStdout()
    with redirect_stdout(dummy_stout):
        p.start()
        result = queue.get()
        p.join()
    dummy_stout.conn.send("DONE")
    return result
    



address = ('0.0.0.0', 6001)     # family is deduced to be 'AF_INET'
listener = Listener(address, authkey=b'secret password')

stdout_listener = Listener(('0.0.0.0', 6002), authkey=b'secret password')

try:
    while True:
        conn = listener.accept()
        print("got new job", listener.last_accepted)
        cell = conn.recv()
        model_name=conn.recv()
        state_file = conn.recv_bytes()
        result = execute_in_subprocess(state_file, cell, model_name, stdout_listener)
        conn.send_bytes(result)
finally:
    listener.close()