"""Bound the work admitted by a single-worker public research demonstration.

These limits are global, not based on untrusted proxy/IP headers. No upload,
filename or client address is retained. This is not a distributed DDoS defence.
"""
from __future__ import annotations

import asyncio
from collections import deque
import time

from starlette.responses import JSONResponse


class PublicAPILimits:
    """Reject excess work before multipart parsing or model execution.

    Only one inference/upload runs at a time. The slot stays occupied until
    execution actually finishes, including after a client disconnect. Native
    CPU operations are not safely interruptible with an asyncio timeout.
    """

    def __init__(self, app, *, max_body_bytes=6 * 1024 * 1024,
                 upload_timeout=15.0, requests_per_minute=12):
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.upload_timeout = upload_timeout
        self.requests_per_minute = requests_per_minute
        self.admitted = deque()
        self.busy = False

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        protected = scope['path'].rstrip('/') in {'/predict', '/explain'}

        async def private_send(message):
            if message['type'] == 'http.response.start':
                message = dict(message)
                headers = list(message.get('headers', []))
                headers.extend([(b'x-content-type-options', b'nosniff'),
                                (b'referrer-policy', b'no-referrer')])
                if protected:
                    headers.append((b'cache-control', b'no-store'))
                message['headers'] = headers
            await send(message)

        async def reject(status, detail, retry=False):
            response = JSONResponse({'detail': detail}, status_code=status,
                                    headers={'Retry-After': '60'} if retry else None)
            await response(scope, receive, private_send)

        # API routes are deliberately exact; encoded slashes/trailing slashes
        # are still protected (ASGI supplies the decoded path).
        if not protected or scope['method'] != 'POST':
            return await self.app(scope, receive, private_send)
        now = time.monotonic()
        while self.admitted and now - self.admitted[0] >= 60:
            self.admitted.popleft()
        if len(self.admitted) >= self.requests_per_minute:
            return await reject(429, 'Research demo rate limit reached; retry in one minute', True)
        if self.busy:
            return await reject(503, 'Research demo is busy; retry shortly', True)
        # No await between testing and setting the single-event-loop flag.
        self.busy = True
        self.admitted.append(now)
        try:
            lengths = [v for k, v in scope.get('headers', []) if k.lower() == b'content-length']
            if lengths:
                try:
                    if len(lengths) != 1 or int(lengths[0]) < 0:
                        raise ValueError
                    if int(lengths[0]) > self.max_body_bytes:
                        return await reject(413, 'Request exceeds the 6 MiB body limit')
                except ValueError:
                    return await reject(400, 'Invalid request length')
            payload = bytearray()

            async def collect():
                while True:
                    message = await receive()
                    if message['type'] == 'http.disconnect':
                        return False
                    block = message.get('body', b'')
                    if len(payload) + len(block) > self.max_body_bytes:
                        raise OverflowError
                    payload.extend(block)
                    if not message.get('more_body', False):
                        return True

            try:
                complete = await asyncio.wait_for(collect(), timeout=self.upload_timeout)
            except asyncio.TimeoutError:
                return await reject(408, 'Upload timed out')
            except OverflowError:
                return await reject(413, 'Request exceeds the 6 MiB body limit')
            if not complete:
                return
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {'type': 'http.request', 'body': bytes(payload), 'more_body': False}
                return await receive()

            await self.app(scope, replay, private_send)
        finally:
            self.busy = False
