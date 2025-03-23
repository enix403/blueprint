import threading
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify

# set path to import minimal
import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parent.parent)

def generate_plan(nodes, edges, scale):
    import time
    time.sleep(5)
    return {"status": "success", "nodes": nodes, "edges": edges, "scale": scale}

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)

tasks = {}
task_lock = threading.Lock()

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        scale = tuple(data.get("scale", ()))
        
        task_id = threading.get_ident()
        
        future = executor.submit(generate_plan, nodes, edges, scale)
        
        with task_lock:
            tasks[task_id] = future
        
        result = future.result()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=3002)
