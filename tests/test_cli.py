"""Tests for command-line interface."""

from click.testing import CliRunner
from spike_transformer_compiler.cli import main, info
from spike_transformer_compiler import __version__


class TestCLI:
    """Test cases for CLI commands."""
    
    def test_main_help(self):
        """Test main command help output."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Spike-Transformer-Compiler" in result.output
        
    def test_version_option(self):
        """Test version option."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output
        
    def test_info_command(self):
        """Test info command."""
        runner = CliRunner()
        result = runner.invoke(info)
        assert result.exit_code == 0
        assert __version__ in result.output
        assert "loihi3" in result.output
        assert "simulation" in result.output
        
    def test_compile_command(self):
        """Test compile command with basic arguments."""
        runner = CliRunner()
        result = runner.invoke(main, ["compile", "dummy_model.pth"])
        assert result.exit_code == 0
        assert "Compiling dummy_model.pth" in result.output
        assert "Implementation pending" in result.output