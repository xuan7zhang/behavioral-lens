"""Local paths for running the exp7 code on this server (GPU 7). Import this before lib / tk_lib."""
import os, sys
os.environ.setdefault("HF_HOME", "/tealab-data/yush/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib
lib.BASE = "/tealab-data/yush/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
lib.M = "/tealab-data/yush/exp7_models"
lib.R = os.path.dirname(HERE)
import tk_lib  # noqa: E402  (its `from lib import *` now sees the local paths)
for k in ("BASE", "M", "R"):
    setattr(tk_lib, k, getattr(lib, k))
NOTES = os.path.join(lib.R, "notes")
FIVE = [(9, 0), (9, 1), (10, 2), (12, 1), (13, 3)]
