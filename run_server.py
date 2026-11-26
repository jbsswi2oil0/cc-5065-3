#!/usr/bin/env python3
"""Launch the photon-estimation experiment web UI.

Usage:
    python3 run_server.py [--host 127.0.0.1] [--port 8000]

Then open the printed URL in a browser.  No third-party packages are needed;
the renderer and the HTTP server use only the Python standard library.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caustics.webapp import make_server


def main():
    ap = argparse.ArgumentParser(description="Photon-mapping caustic workbench")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    httpd = make_server(args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"Photon-estimation workbench running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
