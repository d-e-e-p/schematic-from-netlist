import logging as log
from pathlib import Path
from typing import Optional

import jpype


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
