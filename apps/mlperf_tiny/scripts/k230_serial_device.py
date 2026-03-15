"""K230 serial device adapter for MLPerf Tiny runner.

RT-Smart's msh shell line-buffers stdin, so programs only receive input
after CR is sent.  The upstream SerialDevice sends "command%" but K230
needs "command%\\r" to flush the msh line buffer.

Usage from runner directory:
    # Symlink or copy this file into the runner dir, then:
    from k230_serial_device import K230SerialDevice
    dut = DUT(K230SerialDevice(port, baud, "m-ready", "%"),
              baud_rate=baud)
"""

import sys
import os

# Add runner directory to path so we can import SerialDevice
_RUNNER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "mlperf_tiny", "benchmark", "runner",
)
if _RUNNER_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_RUNNER_DIR))

from interface_device import InterfaceDevice  # noqa: E402
from serial_device import SerialDevice  # noqa: E402


class K230SerialDevice(SerialDevice, InterfaceDevice):
    """SerialDevice that appends CR after the delimiter for msh.

    Also inherits InterfaceDevice so DUT accepts it without re-wrapping.
    """

    def __init__(self, port_device, baud_rate,
                 end_of_response="m-ready", delimiter="%", **kwargs):
        SerialDevice.__init__(self, port_device, baud_rate,
                              end_of_response=end_of_response,
                              delimiter=delimiter, **kwargs)

    def write_line(self, text):
        """Write text + delimiter + CR (msh line buffer flush)."""
        self.write(text)
        self.write(self._delimiter)
        self.write("\r")
