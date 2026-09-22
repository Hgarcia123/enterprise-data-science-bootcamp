"""The project's own code, as a Python package.

This file makes the `src` folder a regular package, so that other files can
write `from src.config import SEED` and the terminal can run
`python -m src.dataset`. It is empty on purpose: it only has to exist.

Python can also import a folder that has no such file, as a "namespace
package". We keep the file anyway, because it is explicit: with it, `src` means
this folder and nothing else. Without it, another folder called `src` found on
Python's import path would be merged into this one.
"""
