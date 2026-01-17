APP_NAME=local-camera-app
STREAMLIT_PORT=8501
APP_FILE=ui/web_ui.py

ENV_FILE=.env
REQUIREMENTS=requirements.txt

install-python:
	pip install --upgrade pip
	pip install -r $(REQUIREMENTS)
	pip install streamlit python-dotenv requests

install-ngrok:
	sudo snap install ngrok

ngrok-auth:
	@if [ ! -f $(ENV_FILE) ]; then echo ".env not found"; exit 1; fi
	@export $$(grep -v '^#' $(ENV_FILE) | xargs) && \
	ngrok config add-authtoken $$NGROK_AUTH_TOKEN

install-gpu:
	sudo apt install -y nvidia-container-toolkit
	sudo systemctl restart docker

install: install-python install-ngrok ngrok-auth

run:
	streamlit run $(APP_FILE) --server.port $(STREAMLIT_PORT) & \
	STREAMLIT_PID=$$!; \
	sleep 5; \
	ngrok http $(STREAMLIT_PORT) > /tmp/ngrok.log 2>&1 & \
	sleep 3; \
	PUBLIC_URL=$$(curl -s http://localhost:4040/api/tunnels | \
		python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"); \
	echo ""; \
	echo "==================== NGROK PUBLIC URL ===================="; \
	echo "$$PUBLIC_URL"; \
	echo "========================================================="; \
	echo ""; \
	wait $$STREAMLIT_PID


clean:
	rm -f /tmp/ngrok.log

help:
	@echo "make install"
	@echo "make install-gpu"
	@echo "make run"
	@echo "make clean"
