"""Command-line entry points: ``field-intel synth | run | ask | actions | serve``."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from sqlalchemy import select

from field_intel.agent.llm import build_gateway
from field_intel.agent.loop import run_agent
from field_intel.config import get_settings
from field_intel.db import ActionRequest, Database
from field_intel.pipeline.run import run_pipeline
from field_intel.pipeline.synthetic import generate_demo

app = typer.Typer(help="Field Intel: satellite -> per-field stress signals -> governed agent.")


@app.command()
def synth(
    out_dir: Annotated[Path, typer.Option(help="Where to write fields.geojson + scenes")] = Path(
        "data/local"
    ),
) -> None:
    """Generate the deterministic synthetic demo dataset (no Sentinel Hub needed)."""
    paths = generate_demo(out_dir)
    for key, path in paths.items():
        typer.echo(f"{key:>12}  {path}")


@app.command()
def run(
    fields: Annotated[Path, typer.Option(exists=True, help="GeoJSON of field polygons")],
    raster: Annotated[Path, typer.Option(exists=True, help="Multispectral GeoTIFF")],
    acquired: Annotated[str, typer.Option(help="Acquisition date YYYY-MM-DD")],
    principal: str = "cli",
) -> None:
    """Ingest fields + compute per-field NDVI for one scene (idempotent)."""
    settings = get_settings()
    db = Database(settings.database_url)
    result = run_pipeline(
        db,
        fields_path=fields,
        raster_path=raster,
        acquired_on=date.fromisoformat(acquired),
        principal=principal,
        stress_threshold=settings.stress_ndvi_threshold,
    )
    typer.echo(json.dumps(result.as_dict(), indent=2))


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question for the agent")],
    principal: str = "cli:grower",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Print tool trace")] = False,
) -> None:
    """Ask the agent a question about field health."""
    settings = get_settings()
    db = Database(settings.database_url)
    llm = build_gateway(
        settings.provider_order,
        anthropic_api_key=settings.anthropic_api_key,
        anthropic_model=settings.anthropic_model,
        anthropic_base_url=settings.anthropic_base_url,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
        openai_base_url=settings.openai_base_url,
        timeout=settings.llm_timeout_seconds,
    )
    with db.session() as session:
        answer = run_agent(
            question,
            session=session,
            llm=llm,
            principal=principal,
            stress_threshold=settings.stress_ndvi_threshold,
            max_steps=settings.agent_max_steps,
            tool_result_max_chars=settings.tool_result_max_chars,
        )
    typer.echo(answer.text)
    typer.echo("")
    typer.echo(
        f"[provider={answer.provider} model={answer.model} steps={answer.steps} "
        f"citations={answer.citations} pending_actions={answer.pending_actions} "
        f"redactions={answer.redactions}]"
    )
    if verbose:
        typer.echo(json.dumps(answer.as_dict()["trace"], indent=2))


@app.command()
def actions(
    status: Annotated[str | None, typer.Option(help="pending | approved | rejected")] = None,
) -> None:
    """List agent-proposed actions and their approval status."""
    settings = get_settings()
    db = Database(settings.database_url)
    with db.session() as session:
        stmt = select(ActionRequest).order_by(ActionRequest.id)
        if status:
            stmt = stmt.where(ActionRequest.status == status)
        for a in session.scalars(stmt).all():
            typer.echo(
                f"#{a.id} {a.status:<9} {a.action:<18} {a.field_id}  evidence={a.evidence}  "
                f"by {a.proposed_by}  {a.reason}"
            )


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Run the FastAPI server."""
    uvicorn.run(
        "field_intel.api.main:create_app", factory=True, host=host, port=port, reload=reload
    )


if __name__ == "__main__":  # pragma: no cover
    app()
