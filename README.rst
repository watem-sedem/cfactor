=======
cfactor
=======

.. image:: https://drone.fluves.net/api/badges/fluves/cfactor/status.svg
    :target: https://drone.fluves.net/fluves/cfactor

This is the documentation of **cfactor**, a Fluves/Marlinks package to...

Installation
=============

To use the package, you can either install the latest release from the
`fluves repo <https://repo.fluves.net/fluves/>`_ using pip OR install the latest version in as development
installation 'from source' to get continuous updates of the current master.

Latest release
--------------

.. warning::
   Installing the latest release using this method is only possible within the
   Fluves/Marlinks network. If you are working from outside the network (or VPN), you should have access to git and
   install the package from source.

Make sure to setup an environment (either virtualenv, conda,....) first with pip available. To install the package,
run the following command:

::

    pip install --upgrade pip
    pip install --extra-index-url https://repo.fluves.net/fluves --extra-index-url https://repo.fluves.net/marlinks cfactor

When all goes well, you have the package installed and ready to use.

Install from source
-------------------

To keep track of the continuous development on the package beyond the major releases, ``git clone`` the
repository somewhere and install the code from source.

Make sure to setup a new environment  in either conda or venv (using tox):

Using conda
^^^^^^^^^^^

When using conda, you can setup the environment using the ``environment.yml`` file included in this repository:

::

    conda env create -f environment.yml

Next, install the package from within the ``cfactor`` folder in the terminal and with the conda environment
activated:

::

    conda activate cfactor
    pip install --no-deps -e .

Using venv
^^^^^^^^^^

Run the ``dev`` tox command, which will create a ``venv`` with a development install of the package and it will register
the environment as a ipykernel (for usage inside jupyter notebook):

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

::

    pre-commit run --all-files

It is a good idea to update the hooks to the latest version:

::

    pre-commit autoupdate

.. _pre-commit: http://pre-commit.com/

Unit testing with pytest
-------------------------

Run the test suite using the ``pytest`` package, from within the main package folder (`cfactor`):

::

    pytest

Or using tox (i.e. in a separate environment)

::

    tox

You will receive information on the test status and the test coverage of the unit tests.

Documentation with sphinx
--------------------------

Build the documentation locally with Sphinx:

::

    tox -e docs

which will create the docs in the ``docs/_build/html`` folder. The ``docs/_build`` directory itself is
left out of version control (and we rather keep it as such ;-)).

In order to get nicely rendered online documentation, we use the ``numpydoc`` format. Check the documentation of the
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ for the detailed specification.
As a minimum, provide the following sections for any public method/class: ``summary``, ``description``, ``parameters``,
``returns`` and ``examples``.

Drone CI
--------

Apart from these tools you can run locally, we use drone continuous integration to run these checks also
on our servers. See https://drone.fluves.net/fluves/cfactor for the results.

The drone provides reports that can be checked:

- The docstring coverage of the functions, see the ``report docstring`` step of the `drone output <https://drone.fluves.net/fluves/cfactor>`_.
- An `interactive unit test coverage report <https://drone-coverage-report.static.fluves.net/cfactor/>`_ with the unit test covered code for each of the files.

https://drone-coverage-report.static.fluves.net/cfactor/

For more information on the initial setup, see the ``README.md`` file in the ``ci`` subfolder.

Package release
===============

The CI will create sdist/wheels and publish these to gitea when git tags are added, making releasing
straight forward. In order to publish a new release, the following steps:

- ``git checkout master, git pull origin master`` (work on up to date master branch)
- Update the ``CHANGELOG.rst`` with the changes for this new release
- ``git commit -m 'Update changelog for release X.X.X' CHANGELOG.rst``
- ``git push origin master``
- Add git tags: ``git tag X.X.X``
- Push the git tags: ``git push X.X.X``

When all test pass, drone CI will publish a pre-release on gitea. To convert this to release:

- On the release page of the repository, draft a new release using the latest git tag
- Copy past the changes from the changelog in the dialog and publish release

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.0.1 and the fluves-extension. For details and usage
information on PyScaffold see https://pyscaffold.org/ and the Fluves extension
see https://git.fluves.net/fluves/pyscaffoldext-fluves/.
