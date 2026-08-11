import asyncio
import json

import typer

from .config import Settings
from .service import GPUService

app = typer.Typer(no_args_is_help=True)
gpu = typer.Typer()
experiment = typer.Typer()
app.add_typer(gpu, name="gpu")
app.add_typer(experiment, name="experiment")


def run(method, *args, **kwargs):
    typer.echo(json.dumps(asyncio.run(method(*args, **kwargs)), default=str, indent=2))


@gpu.command("list")
def gpu_list():
    run(GPUService(Settings()).gpu_list)


@gpu.command("status")
def vast_gpu_status(instance_id: str):
    run(GPUService(Settings()).gpu_status, instance_id)


@experiment.command("status")
def experiment_status(job_id: str):
    run(GPUService(Settings()).experiment_status, job_id)


@experiment.command("logs")
def experiment_logs(job_id: str, tail: int = 200):
    run(GPUService(Settings()).experiment_logs, job_id, tail)
