import os
import time

import docker
import pytest

from tests.performance_tests.performance_test_helpers import (
    delete_file_if_exists,
    store_results_to_csv,
)


@pytest.mark.slow
def test_build_time_performance(docker_img_build_perf_results_path: str) -> None:
    """
    Test and record the build time performance and size of a docker image.

    This function builds a Docker image, measures the build time, gets the image size,
    stores these values in a CSV file, and then removes the Docker image.

    Args:
        docker_img_build_perf_results_path (str): Path to the CSV file where the build
        time performance metrics will be stored.

    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dockerfile_path = os.path.join(script_dir, "../../")

    img_name = "test-image"

    client = docker.from_env()

    # Start recording
    start_time = time.time()

    # build image
    client.images.build(
        path=dockerfile_path,
        tag=img_name,
        nocache=True,
    )

    # Stop recording
    end_time = time.time()
    build_time = end_time - start_time

    # Get Image Size
    image = client.images.get(img_name)
    image_size = image.attrs["Size"] / (1024 * 1024)  # Convert size to MB

    # Store build time performance metrics
    delete_file_if_exists(docker_img_build_perf_results_path)
    store_results_to_csv(
        docker_img_build_perf_results_path,
        ("task", "img_build_time_sec", "img_size_mb"),
        (
            "docker_image_build",
            round(build_time, 4),
            round(image_size, 4),
        ),
    )

    # Remove the Docker image
    client.images.remove(img_name)
