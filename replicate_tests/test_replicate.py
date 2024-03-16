"""
It's tests.
Spins up a cog server and hits it with the HTTP API if you're local; runs through the python client if you're not
"""

import base64
import os
import subprocess
import sys
import time
import pytest
import requests
import replicate
from functools import partial
from PIL import Image
from io import BytesIO
import numpy as np

ENV = os.getenv('TEST_ENV', 'local')
LOCAL_ENDPOINT = "http://localhost:5000/predictions"
MODEL = os.getenv(f'{ENV.upper()}_MODEL', 'no model configured')


def local_run(model_endpoint: str, model_input: dict):
    response = requests.post(model_endpoint, json={"input": model_input})
    data = response.json()

    try:
        datauri = data["output"]
        base64_encoded_data = datauri.split(",")[1]
        data = base64.b64decode(base64_encoded_data)
        return Image.open(BytesIO(data))
    except Exception as e:
        print("Error!")
        print("input:", model_input)
        print(data["logs"])
        raise e


def replicate_run(model: str, version: str, model_input: dict):
    output = replicate.run(
        f"{model}:{version}",
        input=model_input)
    url = output

    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def wait_for_server_to_be_ready(url, timeout=300):
    """
    Waits for the server to be ready.

    Args:
    - url: The health check URL to poll.
    - timeout: Maximum time (in seconds) to wait for the server to be ready.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(url)
            data = response.json()

            if data["status"] == "READY":
                return
            elif data["status"] == "SETUP_FAILED":
                raise RuntimeError(
                    "Server initialization failed with status: SETUP_FAILED"
                )

        except requests.RequestException:
            pass

        if time.time() - start_time > timeout:
            raise TimeoutError("Server did not become ready in the expected time.")

        time.sleep(5)  # Poll every 5 seconds


@pytest.fixture(scope="session")
def inference_func():
    if ENV == 'local':
        return partial(local_run, LOCAL_ENDPOINT)
    elif ENV in {'staging', 'prod'}:
        model = replicate.models.get(MODEL)
        version = model.versions.list()[0]
        return partial(replicate_run, MODEL, version.id)
    else:
        raise Exception(f"env should be local, staging, or prod but was {ENV}")


@pytest.fixture(scope="session", autouse=True)
def service():
    if ENV == 'local':
        print("building model")
        # starts local server if we're running things locally
        build_command = 'cog build -t test-model'.split()
        subprocess.run(build_command, check=True)
        container_name = 'cog-test'
        try:
            subprocess.check_output(['docker', 'inspect', '--format="{{.State.Running}}"', container_name])
            print(f"Container '{container_name}' is running. Stopping and removing...")
            subprocess.check_call(['docker', 'stop', container_name])
            subprocess.check_call(['docker', 'rm', container_name])
            print(f"Container '{container_name}' stopped and removed.")
        except subprocess.CalledProcessError:
            # Container not found
            print(f"Container '{container_name}' not found or not running.")

        run_command = f'docker run -d -p 5000:5000 --gpus all --name {container_name} test-model '.split()
        process = subprocess.Popen(run_command, stdout=sys.stdout, stderr=sys.stderr)

        wait_for_server_to_be_ready("http://localhost:5000/health-check")

        yield
        process.terminate()
        process.wait()
        stop_command = "docker stop cog-test".split()
        subprocess.run(stop_command)
    else:
        yield


def test_base_example(inference_func):
    test_example = {
        'image': 'https://replicate.delivery/pbxt/IS6z50uYJFdFeh1vCmXe9zasYbG16HqOOMETljyUJ1hmlUXU/keanu.jpeg',
        'scale': 4,
        'face_enhance': True
    }

    expected_url = 'https://replicate.delivery/pbxt/lv0iOW3u6DrNOd30ybfmufqWebiuW10YjILw05YZGbeipZZCB/output.png'
    resp = requests.get(expected_url)
    expected_img = Image.open(BytesIO(resp.content))
    img_out = inference_func(test_example)
    assert img_out.size == expected_img.size
    img_out.save('image_out.png')
    expected_img.save('expected_out.png')

    img_out_array = np.array(img_out, dtype=np.uint16)
    expected_array = np.array(expected_img, dtype=np.uint16)
    assert np.allclose(img_out_array, expected_array, atol=20)
