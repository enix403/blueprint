import os
import sys
import time
import platform
from pathlib import Path

from flask import Flask, request, jsonify, Blueprint, current_app
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent.parent))  # nopep8
from minimal.lib import generate_plan  # nopep8

core_bp = Blueprint('core_bp', __name__)


@core_bp.get("/")
def index():
    return 'Welcome', 200


def validate_layout(
    node_types,
    edges,
):
    if len(node_types) < 1:
        raise Exception("No rooms provided as input")

    if len(edges) < 1:
        raise Exception("No edges (doors) provided as input")

    for n in node_types:
        if n < 0:
            raise Exception("Invalid room type")

    for a, b in edges:
        if a < 0 or b < 0:
            raise Exception("Invalid edge")

    return True


executor = ThreadPoolExecutor(max_workers=4)


@core_bp.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        node_types = data.get("node_types", [])
        edges = data.get("edges", [])
        edges = [tuple(e) for e in edges]
        scale = tuple(data.get("scale", ()))

        try:
            if validate_layout(node_types, edges):
                print("[OK] Layout is clean")
        except Exception as e:
            print("[ERR] " + str(e))

        print("Generating layout")
        print("    Rooms: " + str(len(node_types)))

        try:
            future = executor.submit(generate_plan, node_types, edges, scale)
            result = future.result()
        except Exception as e:
            print("[ERR] Failed to generate plan")
            print(e)

        print("Plan generated")

        return jsonify(result)

    except Exception as e:
        raise e


start_time = time.time()


@core_bp.route('/healthcheck', methods=['GET'])
def healthcheck():
    uptime_seconds = time.time() - start_time

    return jsonify({
        "status": "ok",
        "app": current_app.name,
        "uptime_seconds": round(uptime_seconds, 2),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "cpu_count": os.cpu_count(),
        "worker_threads": executor._max_workers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }), 200


def create_app():
    app = Flask(__name__)

    app.register_blueprint(core_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=3002)
