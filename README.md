# Blueprint Microservice

Blueprint is a Python microservice that provides inference capabilities to the FrameCraft REST API. It runs as a lightweight HTTP service using Flask and Gunicorn.

---

## 🚀 Getting Started

### 1. Set up a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the service

- Run the service in development mode using

  ```bash
  python httpservice/httpservice.py
  ```

- Or run it in production mode

  ```bash
  gunicorn -w 4 -b 0.0.0.0:3002 httpservice.httpservice:app
  ```

The service will be accessible at [http://localhost:3002](http://localhost:3002)
