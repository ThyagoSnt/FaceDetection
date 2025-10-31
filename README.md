# Security Face Detection Endpoint  
A RESTful API endpoint for face detection in security applications

## Description  
This project implements a lightweight service for detecting and analyzing human faces in images. It is designed as an endpoint that can be integrated into security workflows (such as access control, surveillance, or authentication). The service loads an underlying face-detection model and exposes simple shell scripts and deployment utilities for easy setup.

## Features  
- Python-based REST API (using `app.py`) for face detection.  
- Modular detection logic contained in `modal_face_detector.py` and utilities in `modal_face_utilities.py`.  
- Shell scripts for installation (`install.sh`), execution (`run.sh`), and deployment (`deploy.sh`).  
- Environment configuration via `.env` file.  
- Ready to deploy in container or virtualized environment (with `requirements.txt`).

## Table of Contents  
- [Getting Started](#getting-started)  
- [Installation](#installation)  
- [Usage](#usage)  
- [API Reference](#api-reference)  
- [Configuration](#configuration)  
- [Deployment](#deployment)  
- [Contributing](#contributing)  
- [License](#license)

## Getting Started  
### Prerequisites  
- Python 3.8+  
- pip (or venv)  
- Unix-like shell (for provided scripts)  
- Optional: Docker (if containerizing the service)  

### Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/ThyagoSnt/security-face-detection-endpoint.git  
   cd security-face-detection-endpoint  
   ```  
2. Create and activate virtual environment (recommended):  
   ```bash
   python3 -m venv venv  
   source venv/bin/activate  
   ```  
3. Install dependencies:  
   ```bash
   ./install.sh  
   ```  
   or  
   ```bash
   pip install -r requirements.txt  
   ```  
4. Create a copy of `.env` and configure environment variables as needed:  
   ```bash
   cp .env.template .env  
   # then edit .env  
   ```

## Usage  
To start the API server:  
```bash
./run.sh  
```  
After the server is running, send HTTP requests to the specified endpoint (e.g., `POST /detect_face`) with an image payload to receive face detection results (bounding boxes, confidence scores, etc.).

## API Reference  
**Endpoint**: `/detect_face` (example)  
**Method**: `POST`  
**Request body** (JSON / multipart):  
```json
{
  "image": "<base64-encoded image>"  
}
```  
**Response** (JSON):  
```json
{
  "faces": [
    {
      "bbox": [x, y, width, height],
      "confidence": 0.95
    }
  ],
  "meta": {
    "num_faces": 2,
    "timestamp": "2025-10-31T12:34:56Z"
  }
}
```  
> *Note*: Modify the paths, parameters and payload format according to your actual implementation in `app.py`.

## Configuration  
Environment variables (in `.env`):  
- `MODEL_PATH` – Path to the face detection model file.  
- `THRESHOLD_CONFIDENCE` – Minimum confidence score to consider a detection valid.  
- `HOST` – Host address to bind the web server.  
- `PORT` – Port number for the API service.  
- `LOG_LEVEL` – Logging level (e.g., INFO, DEBUG).  

## Deployment  
Use `deploy.sh` for a basic deployment routine (pull dependencies, set environment variables, start service). For containerized deployment:  
1. Create a `Dockerfile` (if not present).  
2. Build the container:  
   ```bash
   docker build -t face-detection-endpoint .  
   ```  
3. Run the container:  
   ```bash
   docker run -d -p 8000:8000 -v /local/models:/app/models face-detection-endpoint  
   ```  
4. Monitor logs and scale as needed.

## Contributing  
Contributions are welcome! Please follow these guidelines:  
- Fork the repository and create a new branch for your feature or fix.  
- Ensure code is well-documented and comments are written in English.  
- Add or update tests for any new functionality.  
- Submit a pull request with a detailed description of changes.  

## License  
This project is released under the [MIT License](LICENSE). (Or insert whichever license you intend.)