#!/bin/bash

/home/charlie/school/code/csci5707/webapp/venv/bin/gunicorn -w 4 -b 0.0.0.0:8000 server:app
