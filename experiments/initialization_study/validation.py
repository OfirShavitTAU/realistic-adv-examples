"""Shared input validation and planning errors."""


class PlanError(ValueError):
    """An invalid configuration or experiment group."""


################## VALIDATION HELPERS ##################

"""
    Checks a condition and stops planning if it is false.
    @type condition: bool
    @param condition: The condition required for valid input.
    @type message: str
    @param message: The error message for invalid input.
    @raises PlanError: If the condition is false.
"""
def require(condition: object, message: str) -> None:
    if not condition:
        raise PlanError(message)


"""
    Checks that a name can be used in the planned output folders.
    @type value: str
    @param value: The family, group, or image stem to validate.
    @rtype: str
    @returns: The validated name.
    @raises PlanError: If the name contains a path or invalid characters.
"""
def folder_name(value: str) -> str:
    valid_name = isinstance(value, str) and value not in {"", ".", ".."}
    require(valid_name, "Invalid folder name.")

    require(not any(character in value for character in '/\\:<>"|?*'), "Use folder names, not paths: " + value)
    require(not value.endswith((" ", ".")), "Invalid Windows folder name: " + value)

    return value
