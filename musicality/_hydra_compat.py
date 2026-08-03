"""Workaround for facebookresearch/hydra#3121.

Python 3.14 added eager help-string validation to argparse, which breaks
Hydra's lazily-rendered `--shell-completion` help (a `LazyCompletionHelp`
object, not a str). No released Hydra version fixes this yet. Disabling the
eager check restores the old behavior: help text is still expanded lazily,
and correctly, whenever it's actually printed.
"""

import argparse

if hasattr(argparse.ArgumentParser, "_check_help"):
    argparse.ArgumentParser._check_help = lambda self, action: None
