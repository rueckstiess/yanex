"""Shared query parsing for the experiment list and graph endpoints."""

from ..cli.error_handling import CLIErrorHandler


def parse_filters(tags, started_after, started_before, ended_after, ended_before):
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else None
    after, before, ended_after_dt, ended_before_dt = CLIErrorHandler.parse_time_filters(
        started_after, started_before, ended_after, ended_before
    )
    return tag_list, after, before, ended_after_dt, ended_before_dt
