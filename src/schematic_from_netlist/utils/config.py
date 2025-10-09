import logging as log
import os
from pathlib import Path
from typing import Optional

import colorlog
import jpype


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """
    Set up logging with color output to console and plain output to file.
    After calling this, other files can use `import logging as log` and do `log.info(...)`.

    Args:
        verbose: If True, sets log level to DEBUG, else INFO.
        log_file: Optional path to log file. Defaults to 'logs/schematic-from-netlist.log'.
    """
    if log_file is None:
        log_file = "logs/schematic-from-netlist.log"

    # Make sure directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Clear existing handlers to prevent duplicates
    root_logger = log.getLogger()
    root_logger.handlers.clear()
    root_logger.propagate = False

    # Console handler with color
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(funcName)s:%(message)s")
    )

    # File handler
    file_handler = log.FileHandler(log_file, mode="w")
    file_handler.setFormatter(
        log.Formatter("%(levelname)s - %(funcName)s - %(name)s - %(message)s")
    )

    # Attach handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Set level
    root_logger.setLevel(log.DEBUG if verbose else log.INFO)


def setup_elk(jvm_path: Optional[str] = None) -> None:
    """
    Initialize the JVM and load ELK Java classes.

    Args:
        jvm_path: Path to JVM library.
        jar_path: Path to ELK JAR file.

    Raises:
        FileNotFoundError: If required files are not found.
    """

    # Set default paths
    PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent

    jar_path = PACKAGE_ROOT / "resources" / "elk-server-0.2.0-all.jar"

    # TODO: use jpype.getDefaultJVMPath()
    if jvm_path is None:
        jvm_path = "/opt/homebrew/Cellar/openjdk/24.0.2/libexec/openjdk.jdk/Contents/Home/lib/libjli.dylib"

    # Validate files exist
    if not Path(jar_path).exists():
        log.error(f"ELK JAR file not found at: {jar_path}")
        raise FileNotFoundError(f"Required JAR file not found: {jar_path}")

    if not Path(jvm_path).exists():
        log.error(f"JLI library not found at: {jvm_path}")
        raise FileNotFoundError(f"Required JLI library not found: {jvm_path}")

    # Start JVM
    jpype.startJVM(jvm_path, "-ea", f"-Djava.class.path={jar_path}", "--enable-native-access=ALL-UNNAMED", convertStrings=False)
    if jpype.isJVMStarted():
        log.info(f"JVM started successfully: {jpype.getJVMVersion()}")
    else:
        log.error(f"Failed to start JVM with JAR: {jar_path}")
