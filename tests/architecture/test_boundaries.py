"""Static module-boundary checks; no model, device or simulation execution."""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "teleoperation"
FORBIDDEN = {
    "inputs": {"retargeting", "learning", "robots", "simulation", "apps"},
    "data": {"retargeting", "learning", "inputs", "simulation", "apps", "robots"},
    "retargeting": {"learning", "inputs", "simulation", "apps"},
    "robots": {"inputs", "learning", "apps"},
    "simulation": {"learning", "inputs", "apps"},
    "learning": {"inputs", "simulation", "apps"},
    "contracts": {"inputs", "data", "retargeting", "learning", "robots", "simulation", "apps", "tools"},
    "tools": {"apps", "inputs", "learning", "simulation"},
}


def module_for(path):
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports_for(module, tree):
    imports = set()
    package = module if module.endswith(("contracts", "inputs", "data", "apps", "robots", "simulation", "learning", "retargeting", "tools", "tasks", "hand", "arm", "cli", "demo", "replay", "diagnostics")) else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
            imports.add(base)
            imports.update(base + "." + item.name for item in node.names if item.name != "*")
        elif isinstance(node, ast.Call):
            function = node.func
            name = function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else ""
            if name in ("__import__", "import_module"):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    imports.add(node.args[0].value)
                elif module.split(".")[1] not in ("apps", "cli"):
                    raise AssertionError(f"Uncontrolled dynamic import in {module}:{node.lineno}")
    return imports


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {module_for(path): (path, ast.parse(path.read_text(encoding="utf-8"))) for path in PACKAGE.rglob("*.py")}
        cls.graph = {module: imports_for(module, tree) for module, (_, tree) in cls.sources.items()}

    def test_direct_and_transitive_imports_respect_boundaries(self):
        for origin in self.sources:
            layer = origin.split(".")[1] if "." in origin else ""
            forbidden = FORBIDDEN.get(layer, set())
            todo = [(origin, [origin])]
            visited = set()
            while todo:
                current, route = todo.pop()
                if current in visited: continue
                visited.add(current)
                for target in self.graph.get(current, ()):
                    if not target.startswith("teleoperation."): continue
                    target_layer = target.split(".")[1]
                    self.assertNotIn(target_layer, forbidden, " -> ".join(route + [target]))
                    if target in self.graph: todo.append((target, route + [target]))

    def test_contracts_only_use_standard_library_and_numpy(self):
        for module, imports in self.graph.items():
            if module.startswith("teleoperation.contracts"):
                for target in imports:
                    top = target.partition(".")[0]
                    self.assertTrue(top in sys.stdlib_module_names or top in ("numpy", "teleoperation"), (module, target))

    def test_simulation_does_not_load_a_hand_model(self):
        forbidden = ("torch", "teleoperation.retargeting.hand.transformer", "teleoperation.retargeting.hand.predictor", "teleoperation.retargeting.hand.checkpoint", "teleoperation.data.checkpoint")
        for module, imports in self.graph.items():
            if module.startswith("teleoperation.simulation"):
                for target in imports:
                    self.assertFalse(any(target == name or target.startswith(name + ".") for name in forbidden), (module, target))

    def test_robots_do_not_accept_human_landmarks(self):
        for module, (path, tree) in self.sources.items():
            if module.startswith("teleoperation.robots"):
                text = path.read_text(encoding="utf-8")
                for name in ("RawHandFrame", "UpperBodyObservation", "CanonicalHandFrame"):
                    self.assertNotIn(name, text, module)

    def test_partial_packing_is_auxiliary_only(self):
        for module, (path, _) in self.sources.items():
            if module != "teleoperation.robots.partial":
                self.assertNotIn("pack_partial_limb_vector", path.read_text(encoding="utf-8"), module)

    def test_cli_dispatches_only_to_apps_and_has_no_device_imports(self):
        from teleoperation.cli.registry import COMMANDS
        for _, _, _, handler in COMMANDS: self.assertTrue(handler.startswith("teleoperation.apps."), handler)
        for module, imports in self.graph.items():
            if module.startswith("teleoperation.cli"):
                for target in imports:
                    self.assertNotIn(target.partition(".")[0], {"torch", "cv2", "mediapipe", "pybullet", "roboticstoolbox"})

    def test_old_entries_and_path_injection_are_retired(self):
        for name in ("retargeting", "model", "teleop", "input_adapters", "envs", "tasks", "scripts", "main.py", "main_train_twohand.py", "main_offline_twohand.py", "inspect_angle_h5.py"):
            self.assertFalse((ROOT / name).exists(), name)
        for module, (path, _) in self.sources.items():
            self.assertNotIn("sys.path.insert", path.read_text(encoding="utf-8"), module)
