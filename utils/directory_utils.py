import os


def assert_existence(*paths, flag='WARNING'):
    for path in paths:
        assert os.path.exists(path), "{}| Assertion failed. File not found: {}".format(flag, path)


def try_makedir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
