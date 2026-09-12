"""Disposable image smoke; no production endpoints, credentials or storage."""
import argparse
import hashlib
import json
import subprocess
import time
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--docker", default="docker")
    args = parser.parse_args()
    docker = [args.docker]
    # This public fixture authenticates only the disposable loopback container.
    key = "photo-ci-fixture-only"
    cid = subprocess.check_output(docker + ["run", "--detach", "--rm",
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
        "--memory", "512m", "--memory-swap", "512m", "--cpus", "1", "--pids-limit", "128",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--env", "INTERNAL_API_KEY=" + key, "-p", "127.0.0.1::8000", args.image], text=True).strip()
    try:
        address = subprocess.check_output(docker + ["port", cid, "8000/tcp"], text=True).strip()
        assert address.startswith("127.0.0.1:") and "\n" not in address
        base = "http://" + address
        for attempt in range(40):
            try:
                with urlopen(base + "/health", timeout=1) as response:
                    assert response.status == 200
                break
            except (URLError, TimeoutError, ConnectionError):
                time.sleep(0.25)
        else:
            raise RuntimeError("Photo container did not become healthy")
        try:
            urlopen(Request(base + "/internal/photos/optimize", data=b"bad"), timeout=3)
        except HTTPError as failure:
            assert failure.code == 401
        else:
            raise AssertionError("Unauthenticated upload was accepted")
        # Exercise the accepted pixel ceiling including alpha, not a tiny-only fixture.
        with Image.new("RGBA", (4000, 4000), (12, 45, 160, 90)) as source:
            output = BytesIO()
            source.save(output, format="PNG")
        raw = output.getvalue()
        request = Request(base + "/internal/photos/optimize", data=raw,
            headers={"X-Internal-API-Key": key, "Content-Type": "application/octet-stream"})
        with urlopen(request, timeout=60) as response:
            stored = response.read()
            assert response.status == 200 and len(stored) <= len(raw)
            assert response.headers["X-Source-Sha256"] == hashlib.sha256(raw).hexdigest()
            assert response.headers["X-Stored-Sha256"] == hashlib.sha256(stored).hexdigest()
            with Image.open(BytesIO(stored)) as actual:
                assert actual.size == (4000, 4000)
                assert actual.convert("RGBA").getchannel("A").getextrema() == (90, 90)
        state = json.loads(subprocess.check_output(docker + ["inspect", cid], text=True))[0]
        assert not state["State"]["OOMKilled"] and state["State"]["Running"]
        assert state["Config"]["User"] == "65532:65532"
        print("Photo runtime: authentication, 16MP RGBA, hashes, dimensions, alpha, 512MiB PASS")
    finally:
        subprocess.run(docker + ["rm", "--force", cid], check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
