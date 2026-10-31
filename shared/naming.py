"""Names the code and the CDK stacks must agree on, kept in one place so the two cannot drift."""


def digest_memory_name(project_name: str, stage: str) -> str:
    """The AgentCore Memory name the foundation stack creates; its id is this name plus a suffix."""
    return f"{project_name}_{stage}_digest_state".replace("-", "_")


def metric_dimensions(project: str, stage: str) -> dict[str, str]:
    """The dimension map every EMF record carries, and the SAME map the CDK alarms are built with.

    Undimensioned records from different deployments collapse into one datapoint: a dev run of 5
    items made prod's Maximum<1 empty-digest alarm green on a day prod shipped nothing, and a dev
    agent failure paged on prod's error alarm. One function owns the dimension NAMES so the alarm
    can never read a different datapoint than the code publishes.

    An empty value is dropped rather than emitted: CloudWatch rejects a record with an empty
    dimension value outright, which would lose the metric instead of mis-filing it."""
    return {name: value for name, value in (("Project", project), ("Stage", stage)) if value}
