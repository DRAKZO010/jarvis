# self_evolver.py
"""
Self‑evolving and self‑learning sandbox module.

Features:
- Executes dynamically generated Python snippets in a restricted environment.
- Loads a set of "evolution rules" that dictate what code is allowed to do.
- Rules can be updated at runtime, enabling the system to adapt while staying inside a safe barrier.
- Provides an easy‑to‑extend API for adding new learning behaviours.

The sandbox is intentionally minimal – only a whitelist of safe built‑ins and modules is exposed.
Any attempt to break out of the sandbox raises a SecurityError.
"""

import ast
import json
import sys
import traceback
from types import MappingProxyType
from typing import Any, Dict, List

class SandboxSecurityError(RuntimeError):
    """Raised when code tries to perform a disallowed operation."""
    pass

# Whitelisted built‑ins
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "str": str,
    "sum": sum,
    "zip": zip,
}

class Sandbox:
    """Execute Python code strings in a tightly controlled namespace.

    The sandbox creates a fresh globals dict for each execution, populated only
    with the whitelisted built‑ins and any modules explicitly allowed by the
    active rule set.  Code is parsed with ``ast`` first; any disallowed AST node
    triggers a ``SandboxSecurityError`` before execution begins.
    """
    def __init__(self, allowed_modules: List[str] = None, extra_globals: Dict[str, Any] = None):
        self.allowed_modules = allowed_modules or []
        self.extra_globals = extra_globals or {}
        self._prepare_base_globals()

    def _prepare_base_globals(self) -> None:
        # Read‑only view of safe built‑ins
        self.base_globals = MappingProxyType({k: v for k, v in SAFE_BUILTINS.items()})
        # Dynamically import whitelisted modules
        for mod_name in self.allowed_modules:
            try:
                __import__(mod_name)
                self.base_globals = {**self.base_globals, mod_name: sys.modules[mod_name]}
            except Exception as e:
                raise SandboxSecurityError(f"Failed to import allowed module '{mod_name}': {e}")

    def _validate_ast(self, node: ast.AST) -> None:
        """Recursively walk the AST and reject unsafe constructs."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                raise SandboxSecurityError("Import statements are prohibited; use allowed_modules whitelist.")
            if isinstance(child, ast.Attribute) and child.attr.startswith('__'):
                raise SandboxSecurityError("Access to dunder attributes is prohibited.")
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id in {"eval", "exec", "open", "compile", "globals", "locals", "vars", "__import__"}:
                    raise SandboxSecurityError(f"Use of dangerous function '{child.func.id}' is prohibited.")
            if isinstance(child, ast.Lambda):
                raise SandboxSecurityError("Lambda expressions are not allowed.")
            if isinstance(child, ast.With):
                raise SandboxSecurityError("With‑statements are not allowed.")
            self._validate_ast(child)

    def run(self, code: str) -> Any:
        """Execute *code* safely and return the value of the optional ``result`` variable.

        The user‑provided snippet may assign to ``result`` to communicate the outcome.
        """
        try:
            parsed = ast.parse(code, mode='exec')
        except SyntaxError as e:
            raise SandboxSecurityError(f"Syntax error in provided code: {e}")
        self._validate_ast(parsed)
        exec_globals = dict(self.base_globals)
        exec_globals.update(self.extra_globals)
        exec_locals: Dict[str, Any] = {}
        try:
            exec(compile(parsed, filename='<sandbox>', mode='exec'), exec_globals, exec_locals)
        except Exception as e:
            tb = traceback.format_exc()
            raise SandboxSecurityError(f"Error during sandbox execution:\n{tb}")
        return exec_locals.get('result')

class EvolutionManager:
    """Loads a JSON rule file and mutates a ``Sandbox`` instance accordingly.

    Example rule schema::

        {
            "allowed_modules": ["math", "random"],
            "extra_globals": {"PI": 3.14159}
        }
    """
    def __init__(self, rule_path: str):
        self.rule_path = rule_path
        self.rules: Dict[str, Any] = {}
        self._load_rules()

    def _load_rules(self) -> None:
        try:
            with open(self.rule_path, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Evolution rule file not found: {self.rule_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in rule file: {e}")

    def apply_rules(self, sandbox: Sandbox) -> None:
        allowed = self.rules.get('allowed_modules', [])
        extra = self.rules.get('extra_globals', {})
        sandbox.allowed_modules = allowed
        sandbox.extra_globals = extra
        sandbox._prepare_base_globals()

    def reload_and_apply(self, sandbox: Sandbox) -> None:
        self._load_rules()
        self.apply_rules(sandbox)

if __name__ == '__main__':
    import pathlib
    default_rule_file = pathlib.Path(__file__).with_name('evolution_rules.json')
    manager = EvolutionManager(str(default_rule_file))
    sandbox = Sandbox()
    manager.apply_rules(sandbox)
    demo_code = ""\nresult = sum(i * i for i in range(5))\n""\n    print('Demo result:', sandbox.run(demo_code))
    # Demonstrate that a forbidden operation is blocked
    try:
        sandbox.run('import os\nresult = os.listdir(".")')
    except SandboxSecurityError as e:
        print('Caught security error as expected:', e)
    print('\nEdit', default_rule_file, 'to evolve the sandbox and re‑run.')
"""
