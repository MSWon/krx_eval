# krx_eval

## Prerequisite 
- If you are using Docker then you don't need to install the followings (go to Option-2 if using Docker)
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

- **Evaluate**
    ```shell
    python krx_eval.py
    ```

### Option-2 (Install with docker)
- **Install**
    ```shell
    docker build --tag krx_eval:0.1.0 .
    ```

- **Evaluate**
    ```shell
    docker run -it --entrypoint bash -v $(pwd):$(pwd) -w $(pwd) krx_eval:0.1.0
    ```

    ```shell
    python krx_eval.py
    ```
