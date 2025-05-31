import os
import sys
import io
import contextlib
import time
import matplotlib.pyplot as plt
from datetime import datetime

class MarkdownReportGenerator:
    """Generator for markdown reports with automated capture of outputs and figures"""
    
    def __init__(self, report_name="report", output_dir="reports"):
        self.output_dir = output_dir
        self.report_name = report_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.full_report_name = f"{self.report_name}_{self.timestamp}"
        self.content = []
        self.figures_dir = os.path.join(output_dir, "figures")
        self.figure_counter = 0
        self.stdout_capture = None
        self.original_stdout = None
        self.section_level = 1
        
        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Initialize with metadata
        self.add_header(f"Experiment Report: {self.report_name}", level=1)
        self.add_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    def start_capture(self):
        """Start capturing stdout"""
        self.stdout_capture = io.StringIO()
        self.original_stdout = sys.stdout
        sys.stdout = self.stdout_capture
        
    def stop_capture(self):
        """Stop capturing stdout and return captured text"""
        if self.stdout_capture is None:
            return ""
        
        sys.stdout = self.original_stdout
        captured = self.stdout_capture.getvalue()
        self.stdout_capture = None
        return captured
    
    def start_section(self, title, level=None):
        """Start a new section in the report"""
        if level is None:
            level = self.section_level
        
        # Add any captured text before starting new section
        self._add_captured_text()
        
        # Add section header
        self.add_header(title, level=level)
    
    def add_header(self, text, level=2):
        """Add a header to the report"""
        header = "#" * level
        self.content.append(f"{header} {text}\n")
    
    def add_text(self, text):
        """Add plain text to the report"""
        self.content.append(f"{text}\n")
    
    def add_code(self, code, language="python"):
        """Add code block to the report"""
        self.content.append(f"```{language}\n{code}\n```\n")
    
    def add_figure(self, fig=None, title=None, width=None, height=None):
        """Add a matplotlib figure to the report"""
        if fig is None:
            fig = plt.gcf()
            
        self.figure_counter += 1
        filename = f"figure_{self.timestamp}_{self.figure_counter:03d}.png"
        filepath = os.path.join(self.figures_dir, filename)
        
        # Ensure figure is properly saved
        try:
            fig.savefig(filepath, bbox_inches="tight")
            print(f"Figure saved to {filepath}")
            
            # Add to markdown with optional title
            if title:
                self.content.append(f"**{title}**\n\n")
                
            img_tag = f"![Figure {self.figure_counter}](figures/{filename})"
            if width and height:
                img_tag = f"<img src='figures/{filename}' width='{width}' height='{height}' />"
            elif width:
                img_tag = f"<img src='figures/{filename}' width='{width}' />"
                
            self.content.append(f"{img_tag}\n\n")
            
            # Close the figure to prevent memory leaks
            plt.close(fig)
            
        except Exception as e:
            self.content.append(f"**Error saving figure: {str(e)}**\n\n")
    
    def _add_captured_text(self):
        """Add any captured stdout text to the report"""
        captured = self.stop_capture()
        if captured.strip():
            self.add_code(captured.rstrip())
        self.start_capture()  # Resume capture
    
    def finalize_report(self):
        """Finalize and save the report"""
        # Add any final captured text
        self._add_captured_text()
        
        # Write report to file
        report_path = os.path.join(self.output_dir, f"{self.full_report_name}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("".join(self.content))
        
        return report_path

def setup_report_hooks(report_generator):
    """Setup matplotlib hooks to automatically capture figures"""
    original_show = plt.show
    
    def show_and_save(*args, **kwargs):
        fig = plt.gcf()
        report_generator.add_figure(fig)
        result = original_show(*args, **kwargs)
        return result
    
    plt.show = show_and_save
    return original_show

def restore_plt_hooks(original_show):
    """Restore original matplotlib show function"""
    plt.show = original_show