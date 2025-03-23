import threading
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify

# set path to import minimal
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from minimal.lib import generate_plan

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)

tasks = {}
task_lock = threading.Lock()

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        node_types = data.get("node_types", [])
        edges = data.get("edges", [])
        edges = list(map(tuple, edges))
        scale = tuple(data.get("scale", ()))
        
        task_id = threading.get_ident()
        
        future = executor.submit(generate_plan, node_types, edges, scale)
        
        with task_lock:
            tasks[task_id] = future
        
        result = future.result()
        return jsonify(result)
        
    except Exception as e:
        raise e
        # return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=3002)
