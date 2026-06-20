import os
import platform
import shutil
import subprocess


def _run(cmd, timeout=3):
    """Run a command safely. Returns stdout text or ''."""
    try:
        if isinstance(cmd, str):
            return subprocess.check_output(
                cmd, shell=True, text=True, stderr=subprocess.DEVNULL, timeout=timeout
            ).strip()
        else:
            return subprocess.check_output(
                cmd, shell=False, text=True, stderr=subprocess.DEVNULL, timeout=timeout
            ).strip()
    except Exception:
        return ""


def _first_line(s: str) -> str:
    s = (s or "").strip()
    return s.splitlines()[0].strip() if s else ""


def _which(name: str) -> str:
    return shutil.which(name) or ""


def retrieve_system_info():
    """
    Returns compact system info for C++ compilation:
      - OS and architecture
      - CPU info (brand, logical cores, SIMD capabilities)
      - Available compilers (clang, g++)
    """
    sysname = platform.system()
    machine = platform.machine() or ""

    # CPU brand
    brand = ""
    if sysname == "Linux":
        brand = _run("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip()
    elif sysname == "Darwin":
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])

    # CPU logical cores
    cores_logical = os.cpu_count() or 0

    # SIMD capabilities (most important for optimization)
    simd = []
    if sysname == "Linux":
        flags = _run("grep -m1 'flags' /proc/cpuinfo | cut -d: -f2")
        if flags:
            fset = set(flags.upper().split())
            for x in ("AVX512F", "AVX2", "AVX", "FMA", "SSE4_2", "NEON"):
                if x in fset:
                    simd.append(x)
    elif sysname == "Darwin":
        feats = (
            (
                _run(["sysctl", "-n", "machdep.cpu.features"])
                + " "
                + _run(["sysctl", "-n", "machdep.cpu.leaf7_features"])
            )
            .upper()
            .split()
        )
        for x in ("AVX512F", "AVX2", "AVX", "FMA", "SSE4_2", "NEON"):
            if x in feats:
                simd.append(x)

    # Compiler detection
    clang = _first_line(_run([_which("clang"), "--version"])) if _which("clang") else ""
    gpp = _first_line(_run([_which("g++"), "--version"])) if _which("g++") else ""

    return {
        "os": {
            "system": sysname,
            "arch": machine,
        },
        "cpu": {
            "brand": brand.strip(),
            "cores_logical": cores_logical,
            "simd": sorted(set(simd)),
        },
        "compilers": {
            "clang": clang,
            "g++": gpp,
        },
    }
