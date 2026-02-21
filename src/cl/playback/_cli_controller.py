"""
CLI controller for recording playback.

Provides an interactive command-line interface for controlling playback:
- Space: Toggle pause/play
- Left/Right arrows: Skip backward/forward 5 seconds
- Shift+Left/Right (Unix) or Ctrl+Left/Right (Windows): Skip backward/forward 1 minute
- g: Go to specific timestamp
- r: Restart from beginning
- q: Quit
"""
from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING

# Platform-specific imports
if sys.platform == 'win32':
    import msvcrt  # type: ignore[import-not-found]
else:
    import select
    import termios
    import tty

if TYPE_CHECKING:
    from ._playback_producer import PlaybackProducer

# Constants
_STATUS_UPDATE_INTERVAL = 0.1  # Update every 100ms
_MM_SS_PARTS            = 2
_HH_MM_SS_PARTS         = 3

def _format_time(frames: int, fps: int) -> str:
    """Format frame count as HH:MM:SS.mmm."""
    total_seconds = frames / fps
    hours   = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:06.3f}"
    return f"{minutes}:{seconds:06.3f}"

def _format_timestamp(timestamp: int, start_timestamp: int, fps: int) -> str:
    """Format absolute timestamp as relative time string."""
    relative_frames = timestamp - start_timestamp
    return _format_time(relative_frames, fps)

# Arrow key escape sequence mappings (Unix)
_ARROW_KEY_MAP = {
    'D': 'left',
    'C': 'right',
    'A': 'up',
    'B': 'down',
}

_SHIFT_ARROW_MAP = {
    'D': 'shift-left',
    'C': 'shift-right',
}

# Windows key code mappings
_WINDOWS_KEY_MAP = {
    b'K': 'left',
    b'M': 'right',
    b'H': 'up',
    b'P': 'down',
    b's': 'shift-left',   # Ctrl+Left
    b't': 'shift-right',  # Ctrl+Right
}

# ============================================================================
# Unix-specific functions
# ============================================================================

def _read_char(fd: int, timeout: float = 0.05) -> str:
    """Read a single character from the file descriptor with timeout."""
    if not select.select([fd], [], [], timeout)[0]:
        return ''
    b = os.read(fd, 1)
    return b.decode('utf-8', errors='replace') if b else ''

def _parse_escape_sequence(fd: int) -> str:
    """Parse an escape sequence after receiving ESC."""
    c2 = _read_char(fd, 0.05)
    if c2 != '[':
        return 'esc'

    c3 = _read_char(fd, 0.05)
    if not c3:
        return 'esc'

    # Simple arrow keys
    if c3 in _ARROW_KEY_MAP:
        return _ARROW_KEY_MAP[c3]

    # Modified keys (Shift+arrow)
    if c3 == '1':
        c4 = _read_char(fd, 0.05)
        if c4 == ';':
            _read_char(fd, 0.05)  # modifier (ignored)
            c6 = _read_char(fd, 0.05)  # direction
            return _SHIFT_ARROW_MAP.get(c6, 'esc')

    return 'esc'

def _get_key_nonblocking_unix(fd: int) -> str | None:
    """Get a key press without blocking (Unix). Returns None if no key pressed."""
    if not select.select([fd], [], [], 0)[0]:
        return None

    b = os.read(fd, 1)
    if not b:
        return None

    c = b.decode('utf-8', errors='replace')

    # Handle escape sequences (arrow keys, etc.)
    if c == '\x1b':  # ESC
        return _parse_escape_sequence(fd)

    return c

def _get_key_nonblocking_windows() -> str | None:
    """Get a key press without blocking (Windows). Returns None if no key pressed."""
    if not msvcrt.kbhit():  # type: ignore[name-defined]
        return None

    first_byte = msvcrt.getch()  # type: ignore[name-defined]

    # Regular key
    if first_byte not in {b'\x00', b'\xe0'}:
        try:
            return first_byte.decode('utf-8', errors='replace')
        except Exception:
            return None

    # Special key (arrow keys, function keys, etc.)
    if msvcrt.kbhit():  # type: ignore[name-defined]
        second_byte = msvcrt.getch()  # type: ignore[name-defined]
        return _WINDOWS_KEY_MAP.get(second_byte, 'esc')

    return 'esc'

if sys.platform == 'win32':
    _get_key_nonblocking = _get_key_nonblocking_windows
else:
    _get_key_nonblocking = _get_key_nonblocking_unix  # type: ignore[assignment]

class PlaybackController:
    """
    Interactive CLI controller for playback.

    Handles keyboard input and controls the PlaybackProducer.
    """

    def __init__(
        self,
        producer         : PlaybackProducer,
        frames_per_second: int,
        duration_frames  : int,
    ):
        self._producer          = producer
        self._fps               = frames_per_second
        self._duration_frames   = duration_frames
        self._running           = False
        self._old_term_settings = None

    def _print_status(self, message: str = "") -> None:
        """Print the current playback status."""
        current  = self._producer.current_timestamp
        start    = self._producer.start_timestamp
        progress = (current - start) / self._duration_frames * 100 if self._duration_frames > 0 else 0
        state    = "⏸ PAUSED" if self._producer.is_paused else "▶ PLAYING"

        current_str = _format_timestamp(current, start, self._fps)
        total_str   = _format_time(self._duration_frames, self._fps)

        # Clear line and print status
        status_line = f"\r\033[K{state}  {current_str} / {total_str}  ({progress:5.1f}%)"
        if message:
            status_line += f"  | {message}"
        print(status_line, end="", flush=True)

    @staticmethod
    def _print_help() -> None:
        """Print help message."""
        # In raw mode, \n doesn't include \r, so we need to explicitly add it
        modifier_key = "Ctrl+←/→" if sys.platform == 'win32' else "Shift+←/→"
        lines = (
            "",
            "╭─────────────────────────────────────────────────────╮",
            "│             Recording Playback Controls             │",
            "├─────────────────────────────────────────────────────┤",
            "│      SPACE            Toggle pause/play             │",
            "│      ←/→              Skip ±5 seconds               │",
            f"│      {modifier_key:<16} Skip ±1 minute                │",
            "│      g                Go to timestamp (prompt)      │",
            "│      r                Restart from beginning        │",
            "│      h/?              Show this help                │",
            "│      q                Quit                          │",
            "╰─────────────────────────────────────────────────────╯",
            "",
        )
        print("\r\n".join(lines), end="\r\n", flush=True)

    def run(self) -> None:
        """Run the interactive controller loop."""
        self._running = True

        # Set terminal to raw mode (Unix only)
        fd = sys.stdin.fileno() if sys.platform != 'win32' else None
        if sys.platform != 'win32' and fd is not None:
            self._old_term_settings = termios.tcgetattr(fd)

        try:
            if sys.platform != 'win32' and fd is not None:
                tty.setraw(fd)

            print("\033[?25l", end="")  # Hide cursor

            self._print_help()
            print("\r\nPress 'h' for help, 'q' to quit.\r\n")
            self._print_status()

            last_status_time = time.time()

            while self._running:
                # Handle input
                if sys.platform == 'win32':
                    key = _get_key_nonblocking()
                else:
                    key = _get_key_nonblocking(fd) if fd is not None else None

                if key:
                    self._handle_key(key)

                # Update status display periodically
                now = time.time()
                if now - last_status_time > _STATUS_UPDATE_INTERVAL:
                    self._print_status()
                    last_status_time = now

                time.sleep(0.01)  # Small sleep to avoid busy loop

        finally:
            # Restore terminal
            print("\033[?25h", end="")  # Show cursor
            print("\r\n")  # Move to next line
            if sys.platform != 'win32' and self._old_term_settings is not None and fd is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._old_term_settings)

    def _handle_key(self, key: str) -> None:
        """Handle a key press."""
        if key == ' ':
            # Toggle pause
            is_paused = self._producer.is_paused
            self._producer.set_paused(not is_paused)
            self._print_status("Resumed" if is_paused else "Paused")

        elif key == 'left':
            # Skip back 5 seconds
            self._producer.seek_relative(-5 * self._fps)
            self._print_status("◀◀ -5s")

        elif key == 'right':
            # Skip forward 5 seconds
            self._producer.seek_relative(5 * self._fps)
            self._print_status("▶▶ +5s")

        elif key == 'shift-left':
            # Skip back 1 minute
            self._producer.seek_relative(-60 * self._fps)
            self._print_status("◀◀ -1min")

        elif key == 'shift-right':
            # Skip forward 1 minute
            self._producer.seek_relative(60 * self._fps)
            self._print_status("▶▶ +1min")

        elif key == 'r':
            # Restart
            self._producer.seek_to(self._producer.start_timestamp)
            self._print_status("⏮ Restarted")

        elif key == 'g':
            # Go to timestamp
            self._prompt_goto()

        elif key in {'h', '?'}:
            # Show help
            self._print_help()
            self._print_status()

        elif key in {'q', '\x03'}:  # q or Ctrl-C
            self._running = False
            print("\r\033[KQuitting...", end="\r\n", flush=True)

        elif key == 'esc':
            # Ignore standalone ESC
            pass

    def _prompt_goto(self) -> None:
        """Prompt user for a timestamp to go to."""
        # Show prompt
        print("\r\033[K", end="")
        print("Go to (MM:SS or SS): ", end="", flush=True)

        # Restore terminal for input (Unix only)
        fd = sys.stdin.fileno() if sys.platform != 'win32' else None
        if sys.platform != 'win32' and fd is not None and self._old_term_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_term_settings)
        print("\033[?25h", end="")  # Show cursor

        try:
            line = input()
            timestamp = self._parse_time_input(line)

            if timestamp is not None:
                target = self._producer.start_timestamp + timestamp
                self._producer.seek_to(target)
                self._print_status(f"Jumped to {_format_time(timestamp, self._fps)}")
            else:
                self._print_status("Invalid time format")

        except (EOFError, KeyboardInterrupt):
            self._print_status("Cancelled")

        finally:
            # Restore raw mode (Unix only)
            print("\033[?25l", end="")  # Hide cursor
            if sys.platform != 'win32' and fd is not None:
                tty.setraw(fd)

    def _parse_time_input(self, text: str) -> int | None:
        """Parse a time input string and return frames. Returns None if invalid."""
        text = text.strip()
        if not text:
            return None

        try:
            # Try MM:SS format
            if ':' in text:
                parts = text.split(':')
                if len(parts) == _MM_SS_PARTS:
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                    return int((minutes * 60 + seconds) * self._fps)
                elif len(parts) == _HH_MM_SS_PARTS:
                    # HH:MM:SS
                    hours   = int(parts[0])
                    minutes = int(parts[1])
                    seconds = float(parts[2])
                    return int((hours * 3600 + minutes * 60 + seconds) * self._fps)
            else:
                # Try just seconds
                seconds = float(text)
                return int(seconds * self._fps)
        except ValueError:
            return None

    def stop(self) -> None:
        """Stop the controller."""
        self._running = False
