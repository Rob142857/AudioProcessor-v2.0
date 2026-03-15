"""
Clean Progress Bar Utilities for GUI Console Output
Provides single-line updating progress bars for long-running operations.
"""

import sys
import time
from typing import Optional, Callable


class ConsoleProgressBar:
    """A clean, single-line progress bar for console/GUI output.
    
    Updates in-place using carriage return for clean display.
    Falls back to simple percentage updates if terminal doesn't support \r.
    """
    
    def __init__(self, 
                 total: int, 
                 description: str = "Processing",
                 width: int = 30,
                 show_percentage: bool = True,
                 show_count: bool = True,
                 show_eta: bool = True):
        """
        Initialize the progress bar.
        
        Args:
            total: Total number of items/steps
            description: Description text shown before the bar
            width: Width of the progress bar in characters
            show_percentage: Whether to show percentage
            show_count: Whether to show item count (x/total)
            show_eta: Whether to show estimated time remaining
        """
        self.total = max(1, total)
        self.description = description
        self.width = width
        self.show_percentage = show_percentage
        self.show_count = show_count
        self.show_eta = show_eta
        
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
        self._last_line_length = 0
        self._finished = False
        
    def update(self, amount: int = 1) -> None:
        """Update progress by the specified amount."""
        self.current = min(self.current + amount, self.total)
        self._render()
        
    def set(self, value: int) -> None:
        """Set progress to a specific value."""
        self.current = min(max(0, value), self.total)
        self._render()
        
    def _render(self) -> None:
        """Render the progress bar to console."""
        # Throttle updates to avoid flooding the output
        now = time.time()
        if now - self.last_update < 0.1 and self.current < self.total:
            return
        self.last_update = now
        
        # Calculate progress
        progress = self.current / self.total
        filled = int(self.width * progress)
        empty = self.width - filled
        
        # Build the bar
        bar = "█" * filled + "░" * empty
        
        # Build the status text
        parts = [f"{self.description}: [{bar}]"]
        
        if self.show_percentage:
            parts.append(f"{progress * 100:5.1f}%")
            
        if self.show_count:
            parts.append(f"({self.current}/{self.total})")
            
        if self.show_eta and self.current > 0:
            elapsed = now - self.start_time
            eta = (elapsed / self.current) * (self.total - self.current)
            if eta > 0 and eta < 3600:  # Only show if < 1 hour
                parts.append(f"ETA: {self._format_time(eta)}")
        
        line = " ".join(parts)
        
        # Clear previous line and print new one
        # Use carriage return to stay on same line
        clear = " " * self._last_line_length
        sys.stdout.write(f"\r{clear}\r{line}")
        sys.stdout.flush()
        
        self._last_line_length = len(line)
        
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}:{secs:02d}"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}:{mins:02d}:00"
    
    def finish(self, message: Optional[str] = None) -> None:
        """Complete the progress bar with an optional final message."""
        if self._finished:
            return
        self._finished = True
        
        self.current = self.total
        elapsed = time.time() - self.start_time
        
        # Final render at 100%
        progress = 1.0
        filled = self.width
        bar = "█" * filled
        
        if message:
            final = f"{self.description}: [{bar}] ✓ {message} ({self._format_time(elapsed)})"
        else:
            final = f"{self.description}: [{bar}] ✓ Complete ({self._format_time(elapsed)})"
        
        # Clear and print final line with newline
        clear = " " * self._last_line_length
        sys.stdout.write(f"\r{clear}\r{final}\n")
        sys.stdout.flush()


class StepProgressBar:
    """Progress bar for discrete steps with named phases."""
    
    def __init__(self, steps: list, description: str = "Progress"):
        """
        Initialize with a list of step names.
        
        Args:
            steps: List of step names, e.g., ["Loading", "Processing", "Saving"]
            description: Overall description
        """
        self.steps = steps
        self.description = description
        self.total = len(steps)
        self.current = 0
        self.start_time = time.time()
        self._last_line_length = 0
        
    def next_step(self, custom_message: Optional[str] = None) -> None:
        """Move to the next step."""
        if self.current < self.total:
            step_name = self.steps[self.current] if not custom_message else custom_message
            self.current += 1
            self._render(step_name)
            
    def _render(self, current_step: str) -> None:
        """Render current progress."""
        progress = self.current / self.total
        width = 20
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        
        line = f"{self.description}: [{bar}] {self.current}/{self.total} - {current_step}"
        
        clear = " " * self._last_line_length
        sys.stdout.write(f"\r{clear}\r{line}")
        sys.stdout.flush()
        self._last_line_length = len(line)
        
    def finish(self, message: str = "Complete") -> None:
        """Complete the progress."""
        elapsed = time.time() - self.start_time
        bar = "█" * 20
        line = f"{self.description}: [{bar}] ✓ {message} ({elapsed:.1f}s)"
        
        clear = " " * self._last_line_length
        sys.stdout.write(f"\r{clear}\r{line}\n")
        sys.stdout.flush()


def progress_wrapper(iterable, description: str = "Processing", total: Optional[int] = None):
    """Wrap an iterable with a progress bar.
    
    Usage:
        for item in progress_wrapper(items, "Loading"):
            process(item)
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = 0
    
    if total == 0:
        # Can't show progress without knowing total
        for item in iterable:
            yield item
        return
    
    bar = ConsoleProgressBar(total, description)
    
    for item in iterable:
        yield item
        bar.update()
    
    bar.finish()


def print_progress(message: str, current: int, total: int, width: int = 30) -> None:
    """Simple one-liner progress update.
    
    Usage:
        print_progress("Downloading", 50, 100)
    """
    progress = current / max(1, total)
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    line = f"\r{message}: [{bar}] {progress*100:5.1f}% ({current}/{total})"
    sys.stdout.write(line)
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()
