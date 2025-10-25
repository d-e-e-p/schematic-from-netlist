import logging as log
import os
from pathlib import Path
from typing import Optional

import colorlog
import jpype

# Define TRACE level (more verbose than DEBUG)
TRACE_LEVEL = 5
log.addLevelName(TRACE_LEVEL, "TRACE")


def trace(self, message, *args, **kwargs):
    """Log a message with TRACE level."""
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)


# Add trace method to Logger class
log.Logger.trace = trace


# Also add trace as a module-level function
def _trace_function(message, *args, **kwargs):
    """Module-level trace function."""
    log.getLogger().trace(message, *args, **kwargs)


# Add trace to the logging module
log.trace = _trace_function


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """
    Set up logging with color output to console and plain output to file.
    Console shows INFO+ (or DEBUG+ if verbose), file captures TRACE+ (all messages).

    After calling this, other files can use:
        import logging as log
        log.trace("Ultra detailed info")
        log.debug("Detailed info")
        log.info("Normal info")

    Args:
        verbose: If True, sets console log level to DEBUG, else INFO.
        log_file: Optional path to log file. Defaults to 'logs/schematic-from-netlist.log'.
    """
    if log_file is None:
        log_file = "logs/schematic-from-netlist.log"

    # Make sure directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure root logger - set to TRACE to allow all messages through
    root_logger = log.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(TRACE_LEVEL)  # Allow TRACE and above

    # Console handler with color - respects verbose flag
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(log.DEBUG if verbose else log.INFO)
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s:%(funcName)s:%(message)s",
            log_colors={
                "TRACE": "cyan",
                "DEBUG": "blue",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )

    # File handler - always captures TRACE and above
    file_handler = log.FileHandler(log_file, mode="w")
    file_handler.setLevel(log.DEBUG)  # Capture everything including TRACE
    file_handler.setFormatter(log.Formatter("%(levelname)s - %(funcName)s - %(name)s - %(message)s"))

    # Attach handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    log.info(f"Logging initialized. Console: {'DEBUG' if verbose else 'INFO'}+, File: TRACE+")


def setup_elk(jvm_path: Optional[str] = None) -> None:
    """
    Initialize the JVM and load ELK Java classes.

    Args:
        jvm_path: Path to JVM library.
        jar_path: Path to ELK JAR file.

    Raises:
        FileNotFoundError: If required files are not found.
    """
    if jpype.isJVMStarted():
        return

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
    jpype.startJVM(
        jvm_path,
        "-ea",
        "--enable-native-access=ALL-UNNAMED",
        "--sun-misc-unsafe-memory-access=allow",
        classpath=jar_path,
    )
    if jpype.isJVMStarted():
        log.info(f"JVM started successfully: {jpype.getJVMVersion()}")
    else:
        log.error(f"Failed to start JVM with JAR: {jar_path}")
