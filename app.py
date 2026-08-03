"""Local development entry point for the Vercel Flask application.

Run ``python app.py`` to open the same Python-only dashboard that Vercel
serves from ``api/index.py``.
"""

from api.index import app


if __name__ == "__main__":
    app.run(debug=True)
