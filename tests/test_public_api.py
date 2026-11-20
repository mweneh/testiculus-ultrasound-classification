import asyncio

import pytest

from src.public_api import PublicAPILimits


def run_request(middleware, *, body=b'', headers=(), receive=None, path='/predict'):
    messages = []

    async def receive_body():
        return {'type': 'http.request', 'body': body, 'more_body': False}

    async def send(message):
        messages.append(message)

    asyncio.run(middleware({'type': 'http', 'path': path, 'method': 'POST',
                            'headers': headers}, receive or receive_body, send))
    return messages


async def ok(scope, receive, send):
    await receive()
    await send({'type': 'http.response.start', 'status': 200, 'headers': []})
    await send({'type': 'http.response.body', 'body': b'OK'})


@pytest.mark.parametrize('headers', [[], [(b'content-length', b'11')]])
def test_body_limit_precedes_parser_with_or_without_content_length(headers):
    async def must_not_run(*args):
        pytest.fail('oversize body reached multipart parser')
    messages = run_request(PublicAPILimits(must_not_run, max_body_bytes=10),
                           body=b'x' * 11, headers=headers)
    assert messages[0]['status'] == 413


def test_streamed_limit_and_upload_deadline():
    async def chunk():
        return {'type': 'http.request', 'body': b'xxxx', 'more_body': True}
    assert run_request(PublicAPILimits(ok, max_body_bytes=10), receive=chunk)[0]['status'] == 413

    async def slow():
        await asyncio.sleep(1)
    assert run_request(PublicAPILimits(ok, upload_timeout=0.01), receive=slow)[0]['status'] == 408


def test_global_rate_limit_cannot_be_bypassed_by_forwarded_address():
    app = PublicAPILimits(ok, requests_per_minute=1)
    assert run_request(app)[0]['status'] == 200
    response = run_request(app, headers=[(b'x-forwarded-for', b'new-address')])
    assert response[0]['status'] == 429
    assert (b'retry-after', b'60') in response[0]['headers']
    assert (b'cache-control', b'no-store') in response[0]['headers']


def test_busy_admission_and_disconnect_release_slot():
    app = PublicAPILimits(ok)
    app.busy = True
    assert run_request(app)[0]['status'] == 503
    app.busy = False

    async def disconnected():
        return {'type': 'http.disconnect'}
    assert run_request(app, receive=disconnected) == []
    assert not app.busy
    assert run_request(app)[0]['status'] == 200


def test_exception_releases_slot_and_health_bypasses_admission():
    async def broken(*args):
        raise RuntimeError('failure')
    app = PublicAPILimits(broken)
    with pytest.raises(RuntimeError):
        run_request(app)
    assert not app.busy
    app.app = ok
    app.busy = True
    assert run_request(app, path='/health')[0]['status'] == 200


@pytest.mark.parametrize('lengths', [[b'-1'], [b'invalid'], [b'2', b'3']])
def test_invalid_or_ambiguous_content_length(lengths):
    app = PublicAPILimits(ok)
    assert run_request(app, headers=[(b'content-length', value) for value in lengths])[0]['status'] == 400
