# LLM Runner


## Preparation

### Python versio
* python >= 3.11

### Install
* Debian
    * check [pre.sh](pre.sh)
    ```
    sh pre.sh
    . ~/.bashrc
    sh requirements.sh
    ```
* Ubuntu
    ```
    sh requirements.sh
    ```

## Run server
```
python run_server.py --host 0.0.0.0 --port 8000 --model_name Qwen/Qwen2-VL-2B-Instruct
```

## Test
```
pytest
```