def load_kernel_source(path):
    try:
        with open(path, 'rb') as source_file:
            source_code = source_file.read()
        return source_code, 0
    except FileNotFoundError:
        return None, -1
