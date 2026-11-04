"""Shared query parsing for the experiment list and graph endpoints."""

import click
from fastapi import HTTPException

from ..cli.error_handling import CLIErrorHandler


def parse_filters(tags, started_after, started_before, ended_after, ended_before):
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else None
    try:
        after, before, ended_after_dt, ended_before_dt = (
            CLIErrorHandler.parse_time_filters(
                started_after, started_before, ended_after, ended_before
            )
        )
    except click.ClickException as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return tag_list, after, before, ended_after_dt, ended_before_dt
