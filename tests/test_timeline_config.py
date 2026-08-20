import json
import os
import tempfile
import unittest

from gpu_steward.timeline.config import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_IDLE_SAMPLE_INTERVAL_SECONDS,
    DEFAULT_SAMPLE_INTERVAL_SECONDS,
    GPUHostConfig,
    TimelineConfig,
    TimelineConfigError,
    default_config_path,
)


class TimelineConfigTests(unittest.TestCase):
    def test_defaults_match_frozen_sampling_contract(self):
        config = TimelineConfig()
        self.assertEqual(60.0, config.sample_interval_seconds)
        self.assertEqual(300.0, config.idle_sample_interval_seconds)
        self.assertEqual(DEFAULT_IDLE_SAMPLE_INTERVAL_SECONDS, config.idle_sample_interval_seconds)
        self.assertEqual(DEFAULT_BACKOFF_SECONDS, config.backoff_seconds)
        self.assertEqual(DEFAULT_SAMPLE_INTERVAL_SECONDS, config.sample_interval_seconds)
        self.assertTrue(default_config_path().endswith(os.path.join(".gpu-steward", "timeline.json")))

    def test_mapping_supports_host_alias_and_disabled_indices(self):
        config = TimelineConfig.from_mapping(
            {
                "hosts": {
                    "AI3": {
                        "host": "gpu.example",
                        "disabled_indices": [2, 2, 0],
                        "project": "My_Paper_3rd",
                    }
                }
            }
        )
        self.assertEqual(("AI3",), tuple(item.name for item in config.hosts))
        self.assertEqual("gpu.example", config.hosts[0].host)
        self.assertEqual((0, 2), config.hosts[0].disabled_gpu_indices)
        self.assertEqual("My_Paper_3rd", config.hosts[0].project)
        self.assertEqual(config.to_mapping(), TimelineConfig.from_json(config.to_json()).to_mapping())

    def test_top_level_disabled_indices_can_be_per_host(self):
        config = TimelineConfig.from_mapping(
            {
                "disabled_gpu_indices": {"AI3": [2]},
                "hosts": [{"name": "AI3", "host": "gpu.example"}],
            }
        )
        self.assertEqual((2,), config.hosts[0].disabled_gpu_indices)

    def test_rejects_unsafe_ssh_target_and_invalid_backoff(self):
        with self.assertRaises(TimelineConfigError):
            GPUHostConfig(name="AI3", host="-oProxyCommand=bad")
        with self.assertRaises(TimelineConfigError):
            TimelineConfig(backoff_seconds=(120, 60))
        with self.assertRaises(TimelineConfigError):
            TimelineConfig(sample_interval_seconds=60, idle_sample_interval_seconds=30)
        with self.assertRaises(TimelineConfigError):
            TimelineConfig.from_mapping({"hosts": [{"name": "AI3", "host": "gpu\nexample"}]})

    def test_load_does_not_embed_unknown_keys_or_command_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "timeline.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"hosts": [{"name": "AI3", "host": "gpu.example", "command": ["secret"]}]}, handle)
            loaded = TimelineConfig.load(path)
            self.assertEqual("gpu.example", loaded.hosts[0].host)
            self.assertNotIn("command", loaded.to_json())


if __name__ == "__main__":
    unittest.main()
