###################### sport_info_retrieval/service_utils.py

def load_config_from_yaml(yaml_file_path: str) -> None:
    """
    Reads configuration from a YAML file and sets environment variables.

    :param yaml_file_path: The file path of the YAML configuration file.
    :type yaml_file_path: str
    """
    import os
    import yaml

    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    for key, value in config.items():
        os.environ[key] = str(value)

def free_gpu_memory():
    """
    Frees up GPU memory by triggering garbage collection and clearing PyTorch's CUDA cache.
    """

    import ctypes
    import torch
    import gc
    libc = ctypes.CDLL("libc.so.6")

    libc.malloc_trim(0)
    torch.cuda.empty_cache()
    gc.collect()

def runtime_logger(func):
    """Decorator to log the runtime of a function."""
    import time
    import logging
    from functools import wraps

    @wraps(func)  # Preserves the metadata of the original function
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time of the function
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time of the function
        runtime = end_time - start_time  # Calculate the runtime
        logging.info(f'Function {func.__name__!r} executed in {runtime:.4f} seconds')  # Log the runtime
        return result  # Return the result of the original function
    return wrapper

def error_logger(func):
    """Decorator to log errors occurred in a function."""
    import time
    import logging
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)  # Execute the function
        except Exception as e:
            logging.error(f'Error in function {func.__name__!r}: {e}', exc_info=True)
            raise  # Re-throw the exception after logging
    return wrapper

def gpu_memory_logger(func):
    """Decorator to log GPU memory usage."""
    from functools import wraps
    import logging
    import GPUtil

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check for available GPUs
        gpus = GPUtil.getGPUs()
        if not gpus:
            logging.info("No GPU found.")
            return func(*args, **kwargs)

        # Monitor pre-execution GPU memory usage
        pre_exec_memory = [gpu.memoryUsed for gpu in gpus]
        result = func(*args, **kwargs)

        # Monitor post-execution GPU memory usage
        post_exec_memory = [gpu.memoryUsed for gpu in GPUtil.getGPUs()]
        
        # Calculate peak memory usage during the function execution
        peak_memory = [post - pre for pre, post in zip(pre_exec_memory, post_exec_memory)]

        # Logged memories
        log_memories = zip(pre_exec_memory, post_exec_memory, peak_memory)
        
        # Log the peak GPU memory usage
        for idx, (pre_exec, post_exec, peak) in enumerate(log_memories):
            logging.info(f"pre_exec memory usage for GPU {idx} during {func.__name__!r}: {pre_exec}MB")
            logging.info(f"post_exec memory usage for GPU {idx} during {func.__name__!r}: {post_exec}MB")
            logging.info(f"peak memory usage for GPU {idx} during {func.__name__!r}: {peak}MB")

        return result
    
    return wrapper
