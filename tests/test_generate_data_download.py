"""The data generator survives one truncated download of its source."""

import http.server
import threading

import pytest

generate_data = pytest.importorskip(
    "paper.scripts.generate_data",
    reason="paper/scripts/ is not shipped in sdist",
)

BODY = b"the\t23135851162\nof\t13151942776\n" * 200


class FlakyHandler(http.server.BaseHTTPRequestHandler):
    """First request sends half the declared body, later ones send it all."""

    requests = 0

    def do_GET(self):  # noqa: N802 (http.server naming)
        type(self).requests += 1
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        if type(self).requests == 1:
            self.wfile.write(BODY[: len(BODY) // 2])
            self.close_connection = True
        else:
            self.wfile.write(BODY)

    def log_message(self, *args):
        pass


def test_truncated_download_is_retried(monkeypatch):
    monkeypatch.setattr(generate_data, "DOWNLOAD_RETRY_SECONDS", 0)
    server = http.server.HTTPServer(("127.0.0.1", 0), FlakyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = "http://127.0.0.1:%d/count_1w.txt" % server.server_port
        assert generate_data.fetch_source(url) == BODY
        assert FlakyHandler.requests == 2
    finally:
        server.shutdown()
        server.server_close()
