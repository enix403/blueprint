# 📘 Blueprint

Blueprint is a Python microservice that provides inference capabilities to the FrameCraft REST API. It runs as a lightweight HTTP service using Flask and Gunicorn.

---

## 🚀 Getting Started

> Make sure you are using Python `3.9.x`. Newer versions are currently not supported. The project is tested on Python `3.9.21`

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

## 📝 Notes

- Ensure system dependencies like `gcc`, `graphviz`, and `libgraphviz-dev` are installed if you're running it outside the container and `pygraphviz` fails to build.
