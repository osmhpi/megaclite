from pathlib import Path
import dill  # pylint: disable=import-outside-toplevel
import torch  # pylint: disable=import-outside-toplevel
import click


@click.command()
@click.argument("state")
@click.argument("cell")
@click.argument("model_name")
@click.argument("output")
def main(state, cell, model_name, output):
    # load state from the users notebook
    dill.load_module(state)
    cell_lines = Path(cell).read_text()
    # train the model
    exec(cell_lines)  # pylint: disable=exec-used
    # save the model
    torch.save(globals()[model_name].state_dict(), output)


if __name__ == "__main__":
    main()
