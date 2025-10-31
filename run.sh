#!/bin/bash

gnome-terminal -- bash -c "streamlit run app.py --server.headless true; exec bash"

sleep 5

gnome-terminal -- bash -c "ngrok http 8501; exec bash"