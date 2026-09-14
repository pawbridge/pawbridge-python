"""Authenticated VM feed / preview DB / GPU bridge; standard library only."""
import errno
import ipaddress
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys

SOCKET_DIRECTORY = Path(".local/run/pawbridge-gpu")


def prepare_socket(home: Path) -> Path:
    """Refuse unknown/active paths; remove only our exact, dead Unix socket."""
    uid = os.getuid()
    if home != home.resolve():
        raise RuntimeError("Home must not contain symlinks")
    parent = home
    for component in SOCKET_DIRECTORY.parts:
        parent = parent / component
        parent.mkdir(mode=0o700, exist_ok=True)
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or info.st_mode & 0o022:
            raise RuntimeError("Unsafe socket directory")
    if stat.S_IMODE(parent.stat().st_mode) != 0o700:
        raise RuntimeError("Socket directory must have mode 0700")
    path = parent / "search.sock"
    try:
        before = path.lstat()
    except FileNotFoundError:
        return path
    if not stat.S_ISSOCK(before.st_mode) or before.st_uid != uid:
        raise RuntimeError("Refusing to replace an unowned socket or other file")
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(1)
        try:
            client.connect(str(path))
        except OSError as error:
            if error.errno != errno.ECONNREFUSED:
                raise RuntimeError("Socket state is not safely recoverable") from None
        else:
            raise RuntimeError("GPU socket is already in use")
    after = path.lstat()
    if (after.st_dev, after.st_ino, after.st_uid, after.st_mode) != (
        before.st_dev, before.st_ino, before.st_uid, before.st_mode
    ):
        raise RuntimeError("Socket changed while checking it")
    path.unlink()
    return path


def service_ip(namespace: str, name: str, port: int) -> str:
    result = subprocess.run(
        ["sudo", "-n", "kubectl", "--kubeconfig=/etc/kubernetes/admin.conf",
         "--request-timeout=10s", "-n", namespace, "get", "service", name, "-o", "json"],
        capture_output=True, text=True, timeout=15, check=True,
    )
    service = json.loads(result.stdout)
    if (service["metadata"]["name"] != name or
            service["metadata"]["namespace"] != namespace or
            service["spec"]["type"] != "ClusterIP" or
            not any(p["port"] == port for p in service["spec"]["ports"])):
        raise RuntimeError("VM service contract mismatch")
    ip = ipaddress.IPv4Address(service["spec"]["clusterIP"])
    if not ip.is_private:
        raise RuntimeError("VM service IP must be private IPv4")
    return str(ip)


def prepare_remote() -> dict:
    if os.getuid() != 1000 or Path.home() != Path("/home/vagrant"):
        raise RuntimeError("VM socket owner does not match proxy UID 1000")
    # Discover first: an unavailable cluster must not change the socket path.
    animal = service_ip("pawbridge", "animal-service", 8081)
    mysql = service_ip("databases", "mysql", 3306)
    path = prepare_socket(Path.home())
    return {"animal": animal, "mysql": mysql, "socket": str(path)}


def ssh_args(home: Path) -> list[str]:
    return [
        "/usr/bin/ssh", "-i", str(home / "pawbridge-ai/ssh/vm-key"),
        "-o", "UserKnownHostsFile=" + str(home / "pawbridge-ai/ssh/known_hosts"),
        "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "IdentitiesOnly=yes", "-o", "ForwardAgent=no",
        "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3", "-o", "ExitOnForwardFailure=yes",
        "-o", "ControlMaster=no", "-o", "ControlPath=none",
    ]


def forward_args(ssh: list[str], host: str, discovered: dict) -> list[str]:
    animal = str(ipaddress.IPv4Address(discovered["animal"]))
    mysql = str(ipaddress.IPv4Address(discovered["mysql"]))
    if not all(ipaddress.ip_address(ip).is_private for ip in (animal, mysql)):
        raise RuntimeError("Unexpected VM address")
    path = "/home/vagrant/.local/run/pawbridge-gpu/search.sock"
    if discovered["socket"] != path:
        raise RuntimeError("Unexpected VM socket path")
    return ssh + [
        "-N", "-L", "127.0.0.1:18082:" + animal + ":8081",
        "-L", "127.0.0.1:13306:" + mysql + ":3306",
        "-R", "127.0.0.1:18091:127.0.0.1:18090",
        "-R", path + ":127.0.0.1:18090", host,
    ]


def main() -> None:
    if sys.argv[1:] == ["--prepare-remote"]:
        print(json.dumps(prepare_remote()))
        return
    if sys.argv[1:]:
        raise RuntimeError("Unexpected arguments")
    address = ipaddress.IPv4Address(os.environ.get("PAWBRIDGE_VM_ADDRESS", "192.168.57.11"))
    if not address.is_private:
        raise RuntimeError("VM address must be private IPv4")
    host = "vagrant@" + str(address)
    ssh = ssh_args(Path.home())
    prepared = subprocess.run(
        ssh + [host, "python3 - --prepare-remote"],
        input=Path(__file__).read_text(), capture_output=True, text=True, timeout=45,
        check=True,
    )
    args = forward_args(ssh, host, json.loads(prepared.stdout))
    print("Starting authenticated VM feed, preview DB and private GPU bridge", flush=True)
    os.execv(args[0], args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError):
        # Never echo captured kubectl/SSH output or credentials into service logs.
        print("VM bridge setup failed; check service availability and socket ownership", file=sys.stderr)
        sys.exit(1)
