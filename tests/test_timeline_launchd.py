import os
import plistlib
import subprocess
import tempfile
import unittest

from gpu_steward.timeline.launchd import (
    DASHBOARD_PORT,
    DEFAULT_DASHBOARD_LABEL,
    DEFAULT_LABEL,
    DashboardLaunchAgentSpec,
    LaunchAgentSpec,
    LaunchdController,
    default_dashboard_plist_path,
    install_dashboard_launch_agent,
    install_launch_agent,
    render_dashboard_plist,
    render_plist,
    uninstall_dashboard_launch_agent,
    uninstall_launch_agent,
)


class TimelineLaunchdTests(unittest.TestCase):
    def test_render_contains_only_fixed_collector_arguments(self):
        payload = render_plist("/usr/local/bin/gpu-steward", "/tmp/timeline.json", "/tmp/timeline.log")
        value = plistlib.loads(payload)
        self.assertEqual(DEFAULT_LABEL, value["Label"])
        self.assertEqual(
            ["/usr/local/bin/gpu-steward", "timeline", "collect-loop", "--config", "/tmp/timeline.json"],
            value["ProgramArguments"],
        )
        self.assertNotIn("StartInterval", value)
        self.assertEqual("Background", value["ProcessType"])
        self.assertTrue(value["LowPriorityIO"])
        self.assertEqual("/tmp/timeline.log", value["StandardErrorPath"])

    def test_render_supports_python_module_mode_when_console_script_is_unavailable(self):
        payload = render_plist(
            "/usr/bin/python3",
            "/tmp/timeline.json",
            "/tmp/timeline.log",
            module_name="gpu_steward.cli",
            pythonpath="/checkout/src",
        )
        value = plistlib.loads(payload)
        self.assertEqual(
            [
                "/usr/bin/python3",
                "-m",
                "gpu_steward.cli",
                "timeline",
                "collect-loop",
                "--config",
                "/tmp/timeline.json",
            ],
            value["ProgramArguments"],
        )
        self.assertEqual({"PYTHONPATH": "/checkout/src"}, value["EnvironmentVariables"])

    def test_render_dashboard_is_separate_fixed_local_service(self):
        payload = render_dashboard_plist(
            "/usr/local/bin/gpu-steward", "/tmp/timeline.json", "/tmp/timeline-dashboard.log"
        )
        value = plistlib.loads(payload)
        self.assertEqual(DEFAULT_DASHBOARD_LABEL, value["Label"])
        self.assertEqual(
            [
                "/usr/local/bin/gpu-steward",
                "timeline",
                "serve",
                "--config",
                "/tmp/timeline.json",
                "--port",
                str(DASHBOARD_PORT),
            ],
            value["ProgramArguments"],
        )
        self.assertTrue(value["RunAtLoad"])
        self.assertTrue(value["KeepAlive"])
        self.assertEqual("Background", value["ProcessType"])
        self.assertEqual("/tmp/timeline-dashboard.log", value["StandardErrorPath"])

    def test_dashboard_spec_rejects_non_contract_port(self):
        with self.assertRaisesRegex(Exception, "fixed at 8765"):
            DashboardLaunchAgentSpec(
                "/usr/local/bin/gpu-steward",
                "/tmp/timeline.json",
                "/tmp/timeline-dashboard.log",
                port=8766,
            )

    def test_install_and_uninstall_are_reversible_and_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "collector.plist")
            spec = LaunchAgentSpec("/usr/local/bin/gpu-steward", "/tmp/timeline.json", "/tmp/timeline.log")
            self.assertEqual(path, install_launch_agent(spec, path))
            self.assertEqual("0o600", oct(os.stat(path).st_mode & 0o777))
            self.assertTrue(uninstall_launch_agent(path))
            self.assertFalse(os.path.exists(path))
            self.assertFalse(uninstall_launch_agent(path))

    def test_dashboard_install_and_uninstall_are_reversible_and_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dashboard.plist")
            spec = DashboardLaunchAgentSpec(
                "/usr/local/bin/gpu-steward",
                "/tmp/timeline.json",
                "/tmp/timeline-dashboard.log",
            )
            self.assertEqual(path, install_dashboard_launch_agent(spec, path))
            self.assertEqual("0o600", oct(os.stat(path).st_mode & 0o777))
            self.assertTrue(uninstall_dashboard_launch_agent(path))
            self.assertFalse(os.path.exists(path))
            self.assertFalse(uninstall_dashboard_launch_agent(path))
            self.assertTrue(
                default_dashboard_plist_path("/tmp/home").endswith(DEFAULT_DASHBOARD_LABEL + ".plist")
            )

    def test_controller_builds_user_scoped_argv_without_shell(self):
        calls = []

        def runner(argv):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, "", "")

        controller = LaunchdController(uid=501, runner=runner)
        controller.start("/tmp/collector.plist")
        controller.stop()
        controller.status()
        self.assertEqual(["launchctl", "bootstrap", "gui/501", "/tmp/collector.plist"], calls[0])
        self.assertEqual(["launchctl", "bootout", "gui/501/{}".format(DEFAULT_LABEL)], calls[1])
        self.assertEqual(["launchctl", "print", "gui/501/{}".format(DEFAULT_LABEL)], calls[2])

    def test_controller_ensure_started_is_idempotent(self):
        calls = []
        status_calls = [
            subprocess.CompletedProcess([], 1, "", "not loaded"),
            subprocess.CompletedProcess([], 0, "", ""),
        ]

        def runner(argv):
            calls.append(argv)
            if argv[1] == "print":
                return status_calls.pop(0)
            return subprocess.CompletedProcess(argv, 0, "", "")

        controller = LaunchdController(label=DEFAULT_DASHBOARD_LABEL, uid=501, runner=runner)
        self.assertTrue(controller.ensure_started("/tmp/dashboard.plist"))
        self.assertFalse(controller.ensure_started("/tmp/dashboard.plist"))
        self.assertEqual(
            [
                ["launchctl", "print", "gui/501/{}".format(DEFAULT_DASHBOARD_LABEL)],
                ["launchctl", "bootstrap", "gui/501", "/tmp/dashboard.plist"],
                ["launchctl", "print", "gui/501/{}".format(DEFAULT_DASHBOARD_LABEL)],
            ],
            calls,
        )

    def test_controller_replace_started_reloads_updated_plist(self):
        calls = []

        def runner(argv):
            calls.append(argv)
            if argv[1] == "bootout":
                return subprocess.CompletedProcess(argv, 1, "", "not loaded")
            return subprocess.CompletedProcess(argv, 0, "", "")

        controller = LaunchdController(label=DEFAULT_DASHBOARD_LABEL, uid=501, runner=runner)
        self.assertTrue(controller.replace_started("/tmp/dashboard.plist"))
        self.assertEqual(
            [
                ["launchctl", "bootout", "gui/501/{}".format(DEFAULT_DASHBOARD_LABEL)],
                ["launchctl", "bootstrap", "gui/501", "/tmp/dashboard.plist"],
            ],
            calls,
        )


if __name__ == "__main__":
    unittest.main()
