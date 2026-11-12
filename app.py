"""
Root Streamlit entrypoint wrapper for SocialPulse AI.
Directs execution to dashboard/app.py.
"""

from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import and execute main dashboard
from dashboard.app import *
