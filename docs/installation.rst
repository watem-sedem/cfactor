.. _installation:

Installation
============

Python is used in this package.

We strongly recommend to make use of the seperate ``cfactor`` Python
environment to install the dependencies (by using ``venv`` or ``conda``,
see :ref:`here <installfromsource>`), so it does not
interfere with other Python installation on your machine.

For now, the only option to use the package is to install it from source. In the future
we forsee an installation via a repository such as pip.

.. _installfromsource:

Install from source
-------------------

To install from source, ``git clone`` the repository somewhere on your local
machine and install the code from source.

Make sure to setup a new environment  in either conda or venv:

Using conda
^^^^^^^^^^^

Make sure you have Python, via
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, installed.
The Python dependencies are handled in the ``environment.yml`` file, so
anybody can recreate the required environment:

::

    conda env create -f environment.yml
    conda activate cfactor

If you which to install the dependencies in a conda environment or your choice,
check out the dependencies in the ``environment.yml``-file.

With your ``cfactor`` environment activated (``conda activate cfactor``),
install the cfactor package:

::

    pip install -e .

Or for development purposes of this package, run following code to install
developer dependencies as well (using pip):

::

    pip install -e .[develop]

Using venv
^^^^^^^^^^

Make sure to have `tox <https://tox.readthedocs.io/en/latest/>`_ available.
Run the ``dev`` tox command, which will create a ``venv`` with a development
install of the package and it will register the environment as a ipykernel
(for usage inside jupyter notebook):

::

    tox -e dev

Development
-----------

Want to contribute code or functionalities to the ``cfactor`` package? Great
and welcome on board! Check out the :ref:`dev-guidelines` to get you up and
running.
