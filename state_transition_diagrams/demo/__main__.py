"""Allow running the demo with ``python -m demo``."""
from demo.app import app

if __name__ == "__main__":
    print("\n  ✦  State Transition Diagrams — Demo App")
    print("     http://127.0.0.1:5050\n")
    app.run(host="127.0.0.1", port=5050, debug=True)
