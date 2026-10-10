"""Streamlit Cloud default entry point bridge.

Delegates execution to app.py so both 'app.py' and 'streamlit_app.py'
work out of the box on Streamlit Community Cloud.
"""

from pathlib import Path
import runpy

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "app.py"
    runpy.run_path(str(target), run_name="__main__")
