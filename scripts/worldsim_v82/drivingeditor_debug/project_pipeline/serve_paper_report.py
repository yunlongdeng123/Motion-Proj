#!/usr/bin/env python3
"""Loopback-only report server with byte-range support for synchronized MP4 seeking."""
import functools
import http.server
import os
from pathlib import Path
import re

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/reports/paper50k_comparison')


class ReportHandler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining = None
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            stream = open(path, 'rb')
        except OSError:
            self.send_error(404, 'File not found')
            return None
        size = os.fstat(stream.fileno()).st_size
        requested = self.headers.get('Range')
        start, end = 0, size - 1
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested.strip())
            if not match or not (match[1] or match[2]):
                stream.close()
                self.send_error(400, 'Only a single byte range is supported')
                return None
            if match[1]:
                start = int(match[1])
                end = min(end, int(match[2])) if match[2] else end
            else:
                start = max(0, size - int(match[2]))
            if start >= size or start > end:
                stream.close()
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return None
        self.send_response(206 if requested else 200)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if requested:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Last-Modified', self.date_time_string(os.fstat(stream.fileno()).st_mtime))
        if Path(path).suffix in ('.json', '.html'):
            self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        stream.seek(start)
        self.remaining = end - start + 1
        return stream

    def copyfile(self, source, destination):
        if self.remaining is None:
            return super().copyfile(source, destination)
        remaining = self.remaining
        try:
            while remaining:
                chunk = source.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                destination.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == '__main__':
    handler = functools.partial(ReportHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 8781), handler)
    server.serve_forever()
