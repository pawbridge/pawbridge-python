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
DATABASE_TARGETS = {
    "mysql": ("mysql", 3306, 13306),
    "postgresql": ("pawbridge-postgresql", 5432, 15432),
}


def database_target(backend: str) -> tuple[str, int, int]:
    if backend not in DATABASE_TARGETS:
        raise RuntimeError("Unknown VM database backend")
    return DATABASE_TARGETS[backend]


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


def prepare_remote(backend: str = "mysql") -> dict:
    service, port, _ = database_target(backend)
    if os.getuid() != 1000 or Path.home() != Path("/home/vagrant"):
        raise RuntimeError("VM socket owner does not match proxy UID 1000")
    # Discover first: an unavailable cluster must not change the socket path.
    animal = service_ip("pawbridge", "animal-service", 8081)
    database = service_ip("databases", service, port)
    path = prepare_socket(Path.home())
    return {"animal": animal, backend: database, "socket": str(path)}


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


def forward_args(ssh: list[str], host: str, discovered: dict, backend: str = "mysql") -> list[str]:
    _, remote_port, local_port = database_target(backend)
    animal = str(ipaddress.IPv4Address(discovered["animal"]))
    database = str(ipaddress.IPv4Address(discovered[backend]))
    if not all(ipaddress.ip_address(ip).is_private for ip in (animal, database)):
        raise RuntimeError("Unexpected VM address")
    path = "/home/vagrant/.local/run/pawbridge-gpu/search.sock"
    if discovered["socket"] != path:
        raise RuntimeError("Unexpected VM socket path")
    return ssh + [
        "-N", "-L", "127.0.0.1:18082:" + animal + ":8081",
        "-L", f"127.0.0.1:{local_port}:{database}:{remote_port}",
        "-R", "127.0.0.1:18091:127.0.0.1:18090",
        "-R", path + ":127.0.0.1:18090", host,
    ]


def main() -> None:
    arguments = sys.argv[1:]
    if arguments == ["--prepare-remote"]:
        print(json.dumps(prepare_remote()))
        return
    if len(arguments) == 2 and arguments[0] == "--prepare-remote":
        print(json.dumps(prepare_remote(arguments[1])))
        return
    if sys.argv[1:]:
        raise RuntimeError("Unexpected arguments")
    backend = os.environ.get("PAWBRIDGE_DATABASE_BACKEND", "mysql")
    database_target(backend)  # Validate before SSH or any remote socket changes.
    address = ipaddress.IPv4Address(os.environ.get("PAWBRIDGE_VM_ADDRESS", "192.168.57.11"))
    if not address.is_private:
        raise RuntimeError("VM address must be private IPv4")
    host = "vagrant@" + str(address)
    ssh = ssh_args(Path.home())
    prepared = subprocess.run(
        ssh + [host, "python3 - --prepare-remote " + backend],
        input=Path(__file__).read_text(), capture_output=True, text=True, timeout=45,
        check=True,
    )
    args = forward_args(ssh, host, json.loads(prepared.stdout), backend)
    print(f"Starting authenticated VM feed, {backend} DB and private GPU bridge", flush=True)
    os.execv(args[0], args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError):
        # Never echo captured kubectl/SSH output or credentials into service logs.
        print("VM bridge setup failed; check service availability and socket ownership", file=sys.stderr)
        sys.exit(1)
