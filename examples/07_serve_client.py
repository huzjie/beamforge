"""Example script (run from repo root or examples/)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamforge.serving.server import create_server
from beamforge.serving.client import BeamClient
import threading

srv = create_server(port=8899)
threading.Thread(target=srv.serve_forever, daemon=True).start()

c = BeamClient()
print("health:", c.health())
print("chat:", c.chat("hello beamforge"))
srv.shutdown()
