#!/usr/bin/env python3
"""Lightweight HTTP server for browser enrichment.

Run this in the background:
  python3 scripts/browser_server.py &

It listens on localhost:9797 and accepts:
  POST /enrich  {url, title, text}  — run enrichment
  GET  /health                      — health check

The bookmarklet (to add to your browser) is printed on first run.
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
import urllib.parse
from datetime import date
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENRICH_SCRIPT = HERE / "browser_enrich.py"
PORT = 9797

BOOKMARKLET_JAVASCRIPT = f"""javascript:(function(){{
  var t=document.title,u=window.location.href;
  var b=document.body.innerText.slice(0,8000);
  fetch('http://localhost:{PORT}/enrich',{{
    method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{url:u,title:t,text:b}})
  }}).then(function(r){{return r.text()}}).then(function(t){{
    var w=window.open('','_blank','width=600,height=500');
    w.document.write('<pre style="font:12px monospace;white-space:pre-wrap">'+t+'</pre>');
    w.document.close();
  }}).catch(function(e){{alert('Browser server not running on port {PORT}');}});
}})();"""


class EnrichHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "port": PORT})
        elif self.path == "/bookmarklet":
            self._respond_html(200, f"""<html><body>
<h2>Browser Enrichment Bookmarklet</h2>
<p>Drag this link to your bookmarks bar:</p>
<p><a href="{BOOKMARKLET_JAVASCRIPT}" 
      onclick="return false"
      style="display:inline-block;padding:8px 16px;background:#333;color:#fff;
             text-decoration:none;border-radius:4px;font:14px sans-serif;">
   🧠 Enrich Vault
</a></p>
<p>Then click it on any page to send to the vault enrichment server.</p>
<p>Server running on port {PORT}.</p>
</body></html>""")
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/enrich":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                self._respond(400, {"error": "invalid json"})
                return

            url = data.get("url", "")
            title = data.get("title", "")
            text = data.get("text", "")

            if not text:
                self._respond(400, {"error": "text field required"})
                return

            # Run the enrichment script
            env = os.environ.copy()
            result = subprocess.run(
                [sys.executable, str(ENRICH_SCRIPT),
                 "--url", url,
                 "--title", title[:200],
                 "--body", "-",
                 "--auto-create"],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=120,
                env=env,
            )

            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")

            if result.returncode != 0:
                self._respond(500, {"error": "enrichment failed", "stderr": stderr})
                return

            self._respond_text(200, stdout)
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, status: int, data: dict):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_html(self, status: int, html: str):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_text(self, status: int, text: str):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[browser-server] {args[0]}", file=sys.stderr)


def main():
    print(f"\n🧠 Browser Enrichment Server")
    print(f"   Listening on http://localhost:{PORT}")
    print(f"   POST /enrich  — send page content")
    print(f"   GET  /health  — health check")
    print(f"   GET  /bookmarklet  — bookmarklet setup page")
    print(f"\n📎 Add this bookmarklet to your browser:")
    print(f"   Visit http://localhost:{PORT}/bookmarklet and drag the link")
    print(f"\n   Or create a bookmark with this URL:\n")
    print(f"   {BOOKMARKLET_JAVASCRIPT[:120]}...")
    print(f"\n   (Full JS shown at /bookmarklet)")
    print(f"\n⏸  Ctrl+C to stop\n")

    server = HTTPServer(("127.0.0.1", PORT), EnrichHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
