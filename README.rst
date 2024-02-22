

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

CI
--

#TODO

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

License
-------

#TODO

Contact
-------

We encourage user to submit question, suggestions and bug reports via
the issues platform on GitHub. In case of other questions, one can mail
to <cn-ws@omgeving.vlaanderen.be>

Powered by
----------

![image](docs/_static/png/DepartementOmgeving_logo.png)

![image](docs/_static/png/KULeuven_logo.png)

![image](docs/_static/png/fluves_logo.png)
