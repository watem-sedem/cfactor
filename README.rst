========
C-factor
========


C-factor
========

#TODO

Get started
-----------
Using conda
^^^^^^^^^^^
When using conda, you can setup the environment using the environment.yml file included in this repository:

::

    conda env create -f environment.yml

Next, install the package from within the cfactor folder in the terminal and with the conda environment activated:

::

    conda activate cfactor
    pip install --no-deps -e .

Using venv
^^^^^^^^^^
Run the dev tox command, which will create a venv with a development install of the package and it will register the environment as a ipykernel (for usage inside jupyter notebook):

::

    tox -e dev

Development
============

Want to contribute code or functionalities to the ``cfactor`` package? Great and welcome on board!

We use a number of development tools to support us in improving the code quality. No magic bullet or free
lunch, but just a set of tools as any craftsman has tools to support him/her doing a better job.

For development purposes using conda, make sure to first run ``pip install -e .[develop]`` environment
to prepare the development environment and install all development tools. (When using ``tox -e dev`` this
is already done).

When starting on the development of the ``cfactor`` package, makes sure to be familiar with the following tools. Do
not hesitate to ask the other developers when having trouble using these tools.

Pre-commit hooks
----------------

To ensure a more common code formatting and limit the git diff, make sure to install the `pre-commit`_ hooks. The
required dependencies are included in the development requirements installed when running ``pip install -e .[develop]``.

.. warning::
   Install the ``pre-commit`` hooks before your first git commit to the package!

::

    pre-commit install

on the main level of the package (``cfactor`` folder, location where the file ``.pre-commit-config.yaml`` is located)

If you just want to run the hooks on your files to see the effect (not during a git commit),
you can use the command at any time:

The C-factor is a measure used in erosion and (overland) sediment modelling to
quantify the effect of crops on soil erosion. It is typically defined in the context of
the RUSLE equation, in which gross erosion for an agricultural parcel is estimated.

Get started
===========

This package make use of `Python` (and a limited number of dependencies such as Numpy).
To install the package make sure to check out the installation instructions and follow
the example in the _Get started section_ of the [package documentation](https://cn-ws.github.io/cfactor)

Code
----
The open-source code can be found on [GitHub](https://github.com/cn-ws/cfactor).

Documentation
-------------

The documentation can be found on the [R-factor documentation
page](https://cn-ws.github.io/cfactor/index.html).

License
-------

This project is licensed under the GNU Lesser Public License v3.0, see
[LICENSE](./LICENSE) for more information.

Contact
-------

We encourage user to submit question, suggestions and bug reports via
the issues platform on GitHub. In case of other questions, one can mail
to <cn-ws@omgeving.vlaanderen.be>

Powered by
----------

![image](docs/_static/png/DepartementOmgeving_logo.png)

![image](docs/_static/png/KULeuven_logo.png)

![image](docs/_static/png/VMM_logo.png)

![image](docs/_static/png/fluves_logo.png)

Note
----
This project has been set up using PyScaffold 4.0.1. For details and
usage information on PyScaffold see <https://pyscaffold.org/>.
