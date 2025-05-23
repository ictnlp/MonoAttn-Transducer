import os
import importlib

# automatically import any Python files in the directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("fs_plugins.models.transducer." + file_name)
