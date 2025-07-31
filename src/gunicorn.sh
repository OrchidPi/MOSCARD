#!/bin/bash
gunicorn --log-file=- -c gunicorn.conf.py 