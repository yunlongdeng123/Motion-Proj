"""CPU dependency-source binding tests; actual numerical parity is audited separately."""
from pathlib import Path
import sys
import tempfile
import unittest
from v44_vehicle_surface import load_surface_helpers


class VehicleSurfaceTests(unittest.TestCase):
    def test_dependencies_are_explicit_and_not_an_ambient_legacy_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); a = root/"first"; b = root/"second"; a.mkdir(); b.mkdir()
            (a/"fit_surface.py").write_text("MARKER='explicit-solver'\n")
            (b/"full_pvd.py").write_text("MARKER='explicit-prior'\n")
            old_path = sys.path.copy(); solver, prior, sources = load_surface_helpers([a, b])
            self.assertEqual(solver.MARKER, "explicit-solver")
            self.assertEqual(prior.MARKER, "explicit-prior")
            self.assertEqual(Path(sources["fit_surface"]["path"]), (a/"fit_surface.py").resolve())
            self.assertEqual(sys.path, old_path)

    def test_missing_or_ambiguous_math_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory)/"a"; b = Path(directory)/"b"; a.mkdir(); b.mkdir()
            with self.assertRaises(ValueError): load_surface_helpers([a])
            for p in (a, b):
                (p/"fit_surface.py").write_text("VALUE=1\n"); (p/"full_pvd.py").write_text("VALUE=2\n")
            with self.assertRaises(ValueError): load_surface_helpers([a, b])


if __name__ == "__main__": unittest.main()
