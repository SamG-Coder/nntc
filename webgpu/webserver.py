"""The static web server for this directory, copied from Basis Universal's webgl/webserver_cross_origin.py.

It serves THE DIRECTORY THIS FILE IS IN, whatever the current one is, so that webgpu/ is the document root
and nothing above it is reachable:

    python webgpu/webserver.py

`cd webgpu && python webserver.py` does the same thing, and so does any other working directory. It used to
serve the current directory - SimpleHTTPRequestHandler's default - which meant that starting it from the
tree root quietly published the whole tree, including every out_*/ directory, under a docstring promising
the opposite. The handler takes a `directory` argument for exactly this, and functools.partial is how a
directory is bound to it for a server that constructs one handler per request.

What was taken: the SimpleHTTPRequestHandler subclass and the no-cache header, which is the one thing
`python -m http.server` lacks. It is NOT what makes the page's Reload return new bytes: every asset fetch
the page makes passes `cache: 'no-store'` itself. What the header is for is the browser's own loads - the
page, main.js, style.css - which no script controls, and which Chrome otherwise caches heuristically. That
is the difference between an edit appearing on a refresh and an edit that seems not to have been made.

What was dropped, and why: the two cross-origin headers (Cross-Origin-Opener-Policy: same-origin and
Cross-Origin-Embedder-Policy: require-corp). They exist to make a page cross-origin isolated so that
SharedArrayBuffer is available to Emscripten's threaded encoder. This page has no WebAssembly, no worker and
no shared memory, and WebGPU has no isolation requirement; carrying them would forbid embedding any
cross-origin resource later for no benefit.

The bind address is 127.0.0.1 rather than every interface: this is a development server on one machine.
"""

import functools
import http.server
import os
import socketserver

class NoCacheRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Dev server: never let the browser cache. The page's own fetches already say `cache: 'no-store'`;
        # this is for the loads the page does not make - index.html, main.js, style.css - so that an edit
        # to any of them is on the screen after a refresh rather than one refresh later.
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8082
handler = functools.partial(NoCacheRequestHandler, directory=HERE)

# 2026-09-21: match http.server.HTTPServer, which sets this and which `python -m http.server` has always
# run with; socketserver.TCPServer leaves it False. Added while chasing a restart that had to wait minutes
# before the same port would bind again. NOT MEASURED TO FIX THAT: a bind/connect/close/rebind test on this
# machine rebound immediately either way, so the cause of that wait is still unidentified and this line is
# only the standard setting, not a demonstrated remedy. On Windows SO_REUSEADDR also lets a SECOND live
# server bind a port this one already holds rather than failing - two servers, requests going to whichever
# - which is worth knowing before copying the line somewhere it matters more than a dev server.
socketserver.TCPServer.allow_reuse_address = True

with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
    print("Serving %s at http://localhost:%d - open that in a browser with WebGPU" % (HERE, PORT))
    httpd.serve_forever()
