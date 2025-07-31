"""Command-line interface for Spike-Transformer-Compiler."""

import click
from . import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """Spike-Transformer-Compiler: Neuromorphic compilation for SpikeFormers."""
    pass


@main.command()
@click.argument("model_path")
@click.option("--target", default="loihi3", help="Target hardware platform")
@click.option("--output", "-o", help="Output directory for compiled model")
@click.option("--optimization-level", "-O", default=2, help="Optimization level (0-3)")
@click.option("--time-steps", default=4, help="Number of time steps")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def compile(model_path, target, output, optimization_level, time_steps, verbose):
    """Compile a PyTorch model to neuromorphic hardware."""
    click.echo(f"Compiling {model_path} for {target}...")
    click.echo("Implementation pending - this is a foundational setup")


@main.command()
def info():
    """Show compiler and hardware information."""
    click.echo(f"Spike-Transformer-Compiler v{__version__}")
    click.echo("Supported targets: loihi3, simulation")
    click.echo("Supported models: SpikeFormer, DSFormer")


if __name__ == "__main__":
    main()