"""Install the experiment-local merge implementation, then run the benchmark."""
import importlib.util
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bench.merging", root / "weighted_merging.py")
module = importlib.util.module_from_spec(spec)
sys.modules["bench.merging"] = module
spec.loader.exec_module(module)
runpy.run_path(str(root / "run_benchmark.py"), run_name="__main__")
