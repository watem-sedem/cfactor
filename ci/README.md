# Drone CI Initial setup

## Activate your repository in drone

Go to [drone.fluves.net](https://drone.fluves.net), search for your repository and click on `activate`.

## Setup credentials in drone

Secrets are defined at organization level with following credentials already predefined, so no specific actions are required.

- To publish package releases to gitea, the `release_token` (see vault > `droneci - gitea release token`) is used.
- To publish your documentation as a website on [docs.fluves.net/](https://docs.fluves.net/), the `rsync_key` (see vault > `droneci - coverage-report-minio`) is used.
- To publish the code coverage as a website on [coverage.fluves.net](https://coverage.fluves.net/), the `minio_secret_key` (see vault > `droneci - docs-rsync-key`) is used.

## Prepare Docker containers for drone CI runs

In order to run the `.drone.yml` CI runs, the required Docker containers need to exist in the
Fluves Docker registry [`registry.fluves.net`](https://registry.fluves.net/v2/_catalog).

The Docker specification is depending on the environment setup and the base template for `venv` and `conda`
are provided. Make sure to follow the Fluves docker name convention for tagging the
images: `registry.fluves.net/drone/PACKAGENAME/ENVIRONMENT-VERSION`, e.g. `registry.fluves.net/drone/cfactor/venv-3.9`.

### Virtualenv

Do build a Docker image for virtualenv, choose a specific Python version. For example, prepare an image for Python 3.9:

```bash
docker build -f Dockerfile.venv -t registry.fluves.net/drone/cfactor/venv-3.9 --build-arg "PYTHON_VERSION=3.9" .
docker push registry.fluves.net/drone/cfactor/venv-3.9
```

If you need to test multiple Python versions, create an image for each Docker container and add a separate pipeline in
the `.drone.yml` file for each version.

### Conda

Using conda, the Python environment is part of the `environment.yml` file:

```bash
docker build -f Dockerfile.conda -t registry.fluves.net/drone/cfactor/conda .
docker push registry.fluves.net/drone/cfactor/conda
```
