# krx_eval

## Prerequisite 
- If you are using Docker then you don't need to install the followings (go to Option-2 or Option-3 if using Docker)
- python 3.9.16
- [poetry (1.4.0)](https://python-poetry.org/docs/)
    ```shell
    export POETRY_VERSION=1.4.0
    export POETRY_NO_INTERACTION=1
    curl -sSL https://install.python-poetry.org | python3 -
    ```
## Installation & Run

### Option-1 (Install with poetry)
- **Install**
    ```shell
    git clone https://github.com/MSWon/krx_eval.git

    cd krx_eval

    poetry install
    ```

- **Evaluate MCQA**
    ```shell
    python krx_eval/evaluate_mcqa.py google/gemma-2-9b-it
    ```

- **Evaluate Longform**
    ```shell
    python krx_eval/generate_response_and_get_bleu.py google/gemma-2-9b-it
    ```

    ```shell
    python krx_eval/evaluate_longform.py gemma-2-9b-it
    ```

### Option-2 (Install with docker)
- **Install**
    ```shell
    docker build --tag krx_eval:0.1.0 .
    ```

- **Evaluate MCQA**
    ```shell
    docker run -it --entrypoint bash -v $(pwd):$(pwd) -w $(pwd) krx_eval:0.1.0
    ```

    ```shell
    python krx_eval/evaluate_mcqa.py google/gemma-2-9b-it
    ```

- **Evaluate Longform**
    ```shell
    docker run -it --entrypoint bash -v $(pwd):$(pwd) -w $(pwd) krx_eval:0.1.0
    ```

    ```shell
    python krx_eval/generate_response_and_get_bleu.py google/gemma-2-9b-it
    ```

    ```shell
    python krx_eval/evaluate_longform.py gemma-2-9b-it
    ```

### Option-3 (Pull from dockerhub) (Recommended)
- docker hub: https://hub.docker.com/r/twosubplace/krx_eval/tags
- **Install**
    ```shell
    docker pull twosubplace/krx_eval:0.2.0
    ```

- **Evaluate MCQA**
    ```shell
    docker run -it --entrypoint bash twosubplace/krx_eval:0.2.0
    ```

    ```shell
    python krx_eval/evaluate_mcqa.py google/gemma-2-9b-it
    ```

- **Evaluate Longform**
    ```shell
    docker run -it --entrypoint bash twosubplace/krx_eval:0.2.0
    ```

    ```shell
    python krx_eval/generate_response_and_get_bleu.py google/gemma-2-9b-it
    ```

    ```shell
    python krx_eval/evaluate_longform.py gemma-2-9b-it
    ```

## Docker attach

```
docker ps -a

CONTAINER ID   IMAGE                            COMMAND   CREATED         STATUS         PORTS     NAMES
9cc37bd809c2   twosubplace/krx_eval:0.2.0       "bash"    2 minutes ago   Up 2 minutes             unruffled_heyrovsky
17e399a8e7ca   twosubplace/krx_lm_train:0.3.0   "bash"    3 days ago      Up 22 hours              funny_kepler
```

- attach to container
```
docker attach 9cc37bd809c2
```