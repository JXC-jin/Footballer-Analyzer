from http.server import ThreadingHTTPServer
import json
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from footballer_analyzer.server import ReviewApp, handler_for
import test_reporting


class ServerTests(unittest.TestCase):
    def test_local_api_frame_and_input_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest = test_reporting.ReportingTests().prepare_run(temporary)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(ReviewApp(temporary)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/runs") as response:
                    self.assertEqual(json.load(response)[0]["run_id"], "run_test")
                with urlopen(base + "/api/runs/run_test/frames/4") as response:
                    frame = cv2.imdecode(np.frombuffer(response.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    self.assertEqual(frame.shape[:2], (48, 64))
                for suffix in ["../manifest.json", "run_test/frames/99"]:
                    with self.assertRaises(HTTPError):
                        urlopen(base + "/api/runs/" + suffix)
                foreign_request = Request(base + "/api/runs/run_test/players", data=b'{}', headers={"Content-Type": "application/json", "Origin": "https://example.com"})
                with self.assertRaises(HTTPError) as raised:
                    urlopen(foreign_request)
                self.assertIn("跨来源", raised.exception.read().decode())
                bad_host_request = Request(base + "/api/runs", headers={"Host": "example.com"})
                with self.assertRaises(HTTPError):
                    urlopen(bad_host_request)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
