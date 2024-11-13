#!/bin/bash
#use loads of workers on port 5000

#find cpu count
workers=$(python -c 'import multiprocessing; print(multiprocessing.cpu_count() * 2 + 1)')

#start gunicorn with number of workers

gunicorn --workers=$workers app:app -b 0.0.0.0:8000
