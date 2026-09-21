"""Host bridge contracts; no VM, GPU, credentials or third-party dependencies."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch, call

spec = importlib.util.spec_from_file_location("vm_bridge", Path(__file__).parents[1] / "deploy/connect_vm.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class SocketRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.path = bridge.prepare_socket(self.home)

    def test_missing_and_dead_socket_are_recoverable(self):
        self.assertEqual(0o700, self.path.parent.stat().st_mode & 0o777)
        with socket.socket(socket.AF_UNIX) as server:
            server.bind(str(self.path))
        self.assertEqual(self.path, bridge.prepare_socket(self.home))
        self.assertFalse(self.path.exists())

    def test_active_socket_is_preserved(self):
        with socket.socket(socket.AF_UNIX) as server:
            server.bind(str(self.path))
            server.listen(1)
            inode = self.path.stat().st_ino
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                bridge.prepare_socket(self.home)
            self.assertEqual(inode, self.path.stat().st_ino)

    def test_regular_file_and_symlink_are_preserved(self):
        self.path.write_text("keep")
        with self.assertRaises(RuntimeError):
            bridge.prepare_socket(self.home)
        self.assertEqual("keep", self.path.read_text())
        self.path.unlink()
        target = self.home / "target"
        target.write_text("keep")
        self.path.symlink_to(target)
        with self.assertRaises(RuntimeError):
            bridge.prepare_socket(self.home)
        self.assertTrue(self.path.is_symlink())
        self.assertEqual("keep", target.read_text())

    def test_unsafe_directory_is_refused(self):
        self.path.parent.chmod(0o777)
        with self.assertRaisesRegex(RuntimeError, "Unsafe"):
            bridge.prepare_socket(self.home)
        self.path.parent.chmod(0o700)

    def test_unknown_socket_connection_failure_is_preserved(self):
        with socket.socket(socket.AF_UNIX) as server:
            server.bind(str(self.path))
        with patch.object(socket.socket, "connect", side_effect=TimeoutError):
            with self.assertRaisesRegex(RuntimeError, "not safely recoverable"):
                bridge.prepare_socket(self.home)
        self.assertTrue(self.path.exists())


class ForwardingTests(unittest.TestCase):
    def test_forwarding_preserves_existing_paths_and_adds_private_socket(self):
        data = {"animal": "10.96.1.2", "mysql": "10.96.2.3", "socket": "/home/vagrant/.local/run/pawbridge-gpu/search.sock"}
        args = bridge.forward_args(["ssh"], "vagrant@192.168.57.11", data)
        self.assertIn("127.0.0.1:18082:10.96.1.2:8081", args)
        self.assertIn("127.0.0.1:13306:10.96.2.3:3306", args)
        self.assertIn("127.0.0.1:18091:127.0.0.1:18090", args)
        self.assertIn(data["socket"] + ":127.0.0.1:18090", args)
        for replacement in ({"animal": "8.8.8.8"}, {"socket": "/tmp/unexpected.sock"}):
            with self.subTest(replacement=replacement), self.assertRaises(RuntimeError):
                bridge.forward_args(["ssh"], "host", data | replacement)

    def test_postgresql_forwarding_is_loopback_only_and_has_no_mysql_dependency(self):
        data = {"animal": "10.96.1.2", "postgresql": "10.96.3.4", "socket": "/home/vagrant/.local/run/pawbridge-gpu/search.sock"}
        args = bridge.forward_args(["ssh"], "vagrant@192.168.57.11", data, "postgresql")
        self.assertIn("127.0.0.1:15432:10.96.3.4:5432", args)
        self.assertIn("127.0.0.1:18082:10.96.1.2:8081", args)
        self.assertIn(data["socket"] + ":127.0.0.1:18090", args)
        self.assertFalse(any("13306" in arg or "3306" in arg for arg in args))
        with self.assertRaises(RuntimeError):
            bridge.forward_args(["ssh"], "host", data | {"postgresql": "8.8.8.8"}, "postgresql")
        with self.assertRaises(KeyError):
            bridge.forward_args(["ssh"], "host", data, "mysql")

    def test_postgresql_discovery_looks_up_only_selected_database_before_socket(self):
        with patch.object(bridge.os, "getuid", return_value=1000), patch.object(bridge.Path, "home", return_value=Path("/home/vagrant")), patch.object(bridge, "service_ip", side_effect=["10.96.1.2", "10.96.3.4"]) as lookup, patch.object(bridge, "prepare_socket", return_value=Path("/home/vagrant/.local/run/pawbridge-gpu/search.sock")):
            result = bridge.prepare_remote("postgresql")
            self.assertEqual("10.96.3.4", result["postgresql"])
            self.assertNotIn("mysql", result)
            self.assertEqual([call("pawbridge", "animal-service", 8081), call("databases", "pawbridge-postgresql", 5432)], lookup.call_args_list)

    def test_selected_database_failure_preserves_socket_and_does_not_fallback(self):
        with patch.object(bridge.os, "getuid", return_value=1000), patch.object(bridge.Path, "home", return_value=Path("/home/vagrant")), patch.object(bridge, "service_ip", side_effect=["10.96.1.2", RuntimeError("unavailable")]) as lookup, patch.object(bridge, "prepare_socket") as prepare:
            with self.assertRaises(RuntimeError):
                bridge.prepare_remote("postgresql")
            self.assertEqual(2, lookup.call_count)
            prepare.assert_not_called()

    def test_main_transmits_validated_backend_and_rejects_unknown_before_ssh(self):
        data = {"animal": "10.96.1.2", "postgresql": "10.96.3.4", "socket": "/home/vagrant/.local/run/pawbridge-gpu/search.sock"}
        with patch.dict(os.environ, {"PAWBRIDGE_DATABASE_BACKEND": "postgresql"}), patch.object(bridge.sys, "argv", ["connect_vm.py"]), patch.object(bridge.subprocess, "run") as run, patch.object(bridge.os, "execv") as execute:
            run.return_value.stdout = json.dumps(data)
            bridge.main()
            self.assertEqual("python3 - --prepare-remote postgresql", run.call_args.args[0][-1])
            self.assertIn("127.0.0.1:15432:10.96.3.4:5432", execute.call_args.args[1])
        with patch.dict(os.environ, {"PAWBRIDGE_DATABASE_BACKEND": "postgresql; false"}), patch.object(bridge.sys, "argv", ["connect_vm.py"]), patch.object(bridge.subprocess, "run") as run:
            with self.assertRaises(RuntimeError):
                bridge.main()
            run.assert_not_called()

    def test_discovery_failure_never_prepares_socket(self):
        with patch.object(bridge.os, "getuid", return_value=1000), patch.object(bridge.Path, "home", return_value=Path("/home/vagrant")), patch.object(bridge, "service_ip", side_effect=RuntimeError), patch.object(bridge, "prepare_socket") as prepare:
            with self.assertRaises(RuntimeError):
                bridge.prepare_remote()
            prepare.assert_not_called()

    def test_service_discovery_requires_expected_identity_and_port(self):
        service = {"metadata": {"name": "animal-service", "namespace": "pawbridge"}, "spec": {"type": "ClusterIP", "clusterIP": "10.96.1.2", "ports": [{"port": 8081}]}}
        with patch.object(bridge.subprocess, "run") as run:
            run.return_value.stdout = json.dumps(service)
            self.assertEqual("10.96.1.2", bridge.service_ip("pawbridge", "animal-service", 8081))
            service["metadata"]["namespace"] = "other"
            run.return_value.stdout = json.dumps(service)
            with self.assertRaises(RuntimeError):
                bridge.service_ip("pawbridge", "animal-service", 8081)


if __name__ == "__main__":
    unittest.main()
