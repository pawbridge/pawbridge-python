# CPU photo optimization service

This entrypoint imports neither Torch nor Elasticsearch. It has no download,
database, R2, or other network client. It returns a storage candidate; the caller
must upload and verify it before marking an APMS photo as archived.

## Run and verify

```sh
python -m pip install -r requirements-photo.txt
uvicorn app.photo_main:app --host 127.0.0.1 --port 8000 --workers 1
# Test dependencies also include httpx==0.27.0.
python -m unittest discover -s tests -p 'test_photo*.py'
docker build -f Dockerfile.photo -t pawbridge-photo:local .
```

Inject `INTERNAL_API_KEY` through the runtime secret mechanism. Do not put keys
in source control, image build arguments, or command history. Keep this service
on an internal network. Use one worker per process/container: the capacity limit
is local, and every replica allows one additional simultaneous operation.

## Contract

`POST /internal/photos/optimize` takes raw image bytes, not JSON, a URL, or
multipart. Supply `X-Internal-API-Key`. The server detects the actual format;
the caller's Content-Type is not trusted. Encoded request bodies are rejected.

Limits: 10 MiB input, 16 million pixels, 15 seconds to receive the request body,
one active read/optimization. The body timeout is not a CPU execution timeout.
Cancellation does not admit a second operation until active CPU work finishes.

Static JPEG, PNG and WebP are accepted. Eligible RGB/RGBA JPEG and PNG are encoded
as WebP quality 90, method 4, only when this yields fewer bytes. There is no crop,
resize or upscale. EXIF orientation is applied to encoded results, and alpha/ICC
are preserved. Existing WebP and other decoded color modes retain the original
bytes to avoid repeated lossy encoding or implicit color/depth conversion.
Animations and invalid images are rejected; encoding failure is retryable.
Lossy compression does not guarantee preservation of every fine identifying mark.

Successful responses contain binary data with its actual Content-Type, no-store
cache policy, and these metadata headers:

| Header | Meaning |
| --- | --- |
| X-Source-Sha256 | SHA-256 of the input bytes |
| X-Stored-Sha256 | SHA-256 of the returned bytes to verify after upload |
| X-Photo-Width / X-Photo-Height | Display dimensions after EXIF orientation, including original fallback |
| X-Photo-Recipe | `original-v1` or `webp-q90-m4-fullsize-v1` |

401 means missing/wrong credentials; 408 is upload timeout; 413 is a byte/pixel
limit; 415 is unsupported HTTP content encoding; 422 is an invalid/unsupported
image; 503 is missing configuration, capacity exhaustion or processing failure.
Busy/processing responses include Retry-After. Callers must use bounded retries
and durable work state, never translate a failed optimization into archive success.

The dedicated `photo-service-image-ci.yml` tests only photo contracts using
`requirements-photo.txt`, builds `Dockerfile.photo`, and exercises the running
image with authentication and a 16MP RGBA fixture under a 512MiB limit. PRs do not
publish. A dev push (or manual dev run with publish=true) publishes the exact
verified image as `dorosiya/pawbridge-photo-service:sha-<revision>` and records its
registry digest. There is no latest tag or automatic Infra change in this workflow.

For the same disposable runtime check locally:

```sh
python .github/scripts/check_photo_image.py pawbridge-photo:local
```

The existing AI image CI remains separate. Pin the photo digest in its own
runtime configuration before rollout; publication does not activate APMS
archiving, modify the database or write R2 objects.
