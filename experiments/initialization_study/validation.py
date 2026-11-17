"""Shared input validation and planning errors."""


class PlanError(ValueError):
    """An invalid configuration or experiment group."""


################## VALIDATION HELPERS ##################

"""
    Raises PlanError when a required condition is false.
"""
def require(condition: object, message: str) -> None:
    if not condition:
        raise PlanError(message)


"""
    Checks that a family, group, or image stem is safe to use in output paths.
    Rejects path separators and names incompatible with Windows.
"""
def folder_name(value: str) -> str:
    valid_name = isinstance(value, str) and value not in {"", ".", ".."}
    require(valid_name, "Invalid folder name.")

    require(not any(character in value for character in '/\\:<>"|?*'), "Use folder names, not paths: " + value)
    require(not value.endswith((" ", ".")), "Invalid Windows folder name: " + value)

    return value
