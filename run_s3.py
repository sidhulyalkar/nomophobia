#!/usr/bin/env python
"""Compatibility shim for `nomophobia s3`."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from s6e8.cli import main
if __name__=='__main__':raise SystemExit(main(['s3',*sys.argv[1:]]))
