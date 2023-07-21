"""This module implements the server for the remote GPU extension."""
from contextlib import redirect_stdout
from io import BytesIO
from multiprocessing import Process, Queue
from multiprocessing.connection import Listener

import click


# pylint: disable=too-few-public-methods
class RemoteStdout:
    """Wraps a multiprocessing.connection.Listener to be file compatible."""

    def __init__(self, listener: Listener) -> None:
        self.conn = listener.accept()

    def write(self, text):
        """Send the text back to the client."""
        self.conn.send(text)


def reexecute_and_train(state_file: str, cell: str, model_name: str, queue):
    """This function runs in the subprocess."""
    # import these modules only in the subprocess
    import dill  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    # load state from the users notebook
    file = BytesIO(state_file)
    dill.load_module(file)

    # train the model
    exec(cell)  # pylint: disable=exec-used

    # save the model
    out_file = BytesIO()
    torch.save(globals()[model_name].state_dict(), out_file)
    queue.put(out_file.getvalue())


def execute_in_subprocess(state_file, cell, model_name, stdout_listener):
    """Setup the subprocess execution with stdout redirect."""

    queue = Queue()
    process = Process(
        target=reexecute_and_train, args=(state_file, cell, model_name, queue)
    )

    remote_stout = RemoteStdout(stdout_listener)
    with redirect_stdout(remote_stout):
        process.start()
        result = queue.get()
        process.join()
    remote_stout.conn.send("DONE")
    return result


@click.command()
@click.option("-h", "--host", default="localhost")
@click.option("-p", "--port", default=6001, type=int)
@click.option("--password", prompt=True, hide_input=True)
def main(host, port, password):
    """The main function"""
    listener = Listener((host, port), authkey=password)
    stdout_listener = Listener((host, port + 1), authkey=password)

    try:
        while True:
            conn = listener.accept()
            print("got new job", listener.last_accepted)
            cell = conn.recv()
            model_name = conn.recv()
            state_file = conn.recv_bytes()
            result = execute_in_subprocess(
                state_file, cell, model_name, stdout_listener
            )
            conn.send_bytes(result)
    finally:
        listener.close()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
