import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from track_particles import config_arguments


class ConfigVectorTest(unittest.TestCase):
    def test_negative_vector_components_round_trip(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "run.toml"
            config.write_text("""
[files]
case = "case.foam"
output = "tracks.csv"
[injection]
initial_velocity = [-0.00001, 0, 1.25]
[particle]
gravity = [0, -5e-324, -1e20]
""")
            parser = argparse.ArgumentParser()
            parser.add_argument("case")
            parser.add_argument("output")
            parser.add_argument("--initial-velocity", nargs=3, type=float)
            parser.add_argument("--gravity", nargs=3, type=float)
            args = parser.parse_args(config_arguments(config))
            self.assertEqual(args.initial_velocity, [-1e-5, 0, 1.25])
            self.assertEqual(args.gravity, [0, -5e-324, -1e20])
