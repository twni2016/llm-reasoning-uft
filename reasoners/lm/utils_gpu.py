import os
import sys
import subprocess


def get_gpu_info():
    """
    Retrieves GPU indices and their available memory using nvidia-smi.

    Returns:
        List[dict]: A list of dictionaries containing GPU index and free memory in MB.
    """
    try:
        # Query GPU index and memory information in CSV format without headers
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as e:
        print("Error executing nvidia-smi:", e.stderr)
        sys.exit(1)

    gpu_info = []
    for line in result.stdout.strip().split("\n"):
        if line:
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    index = int(parts[0].strip())
                    free_mem = float(parts[1].strip())  # in MB
                    gpu_info.append({"index": index, "free_mem": free_mem})
                except ValueError:
                    print(f"Unexpected output format: {line}")
                    continue
    return gpu_info


def select_gpus(num_gpus):
    """
    Selects the top `num_gpus` GPUs with the most free memory.

    Args:
        num_gpus (int): Number of GPUs to select.

    Returns:
        List[int]: List of selected GPU indices.
    """
    gpus = get_gpu_info()
    if not gpus:
        print("No GPUs found on the system.")
        sys.exit(1)

    # Sort GPUs by free memory in descending order
    sorted_gpus = sorted(gpus, key=lambda x: x["free_mem"], reverse=True)

    if num_gpus > len(sorted_gpus):
        print(f"Requested {num_gpus} GPUs, but only {len(sorted_gpus)} available.")
        sys.exit(1)

    selected = [gpu["index"] for gpu in sorted_gpus[:num_gpus]]
    return selected


def set_cuda_visible_devices(num_gpus: int):
    """
    Sets the CUDA_VISIBLE_DEVICES environment variable.
    """
    selected_gpus = select_gpus(num_gpus)
    cuda_devices = ",".join(map(str, selected_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    print(f"Set CUDA_VISIBLE_DEVICES to: {cuda_devices}")
