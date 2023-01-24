#!/usr/bin/env bash

source venv/bin/activate
pip install --upgrade vulture
vulture txt2img.py ldm

