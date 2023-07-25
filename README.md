# MEGACLITE (Jupyter Remote GPU)

This project provides an ipython magic extension, to enable training pytorch models on remote GPU servers.

This project is named `Megaclite` after one of the moons of Jupiter.[1]

[1]: https://en.wikipedia.org/wiki/Megaclite


## Getting Started

~~~bash
# clone the repo
git clone git@gitlab.hpi.de:kisz/research/megaclite.git
cd megaclite
# create a new virtual environment
python3 -m venv venv
source ./venv/bin/activate
# install dependencies and additional dependencies for the demo
pip3 install .[demo]
~~~

Forward port 6001 to a remote server with a running server instance.

~~~bash
ssh -L 6001:127.0.0.1:6001 vm-midea04.eaalab.hpi.uni-potsdam.de
~~~

You should now be able to run the `client.ipynb` notebook.

The first run will probably be quite slow, because the server will recreate your local venv, including compiling the correct python version.