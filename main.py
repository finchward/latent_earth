"""
main.py
Entry point — run with:  python main.py
"""

import threading
import time
import webbrowser

import uvicorn

from config import PORT

if __name__ == "__main__":
    def _open_browser():
        time.sleep(1.8)
        webbrowser.open(f"http://localhost:{PORT}")

    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=PORT,
        log_level="warning",  # startup progress goes via state.log()
    )
