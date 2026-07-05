import re

FEATURE_MESSAGES = {
    "I_A_rms": "Current RMS on phase A is unusually high",
    "I_B_rms": "Current RMS on phase B is unusually high",
    "I_C_rms": "Current RMS on phase C is unusually high",
    "V_A_rms": "Voltage RMS on phase A changed significantly",
    "V_B_rms": "Voltage RMS on phase B changed significantly",
    "V_C_rms": "Voltage RMS on phase C changed significantly",
}

CONTEXT_MESSAGES = {
    "phase_select": "The faulted phase selection contributed to this prediction",
    "fault_resistance": "The estimated fault resistance contributed to this prediction",
    "sc_location": "The estimated fault location along the line contributed to this prediction",
}

STAT_DESCRIPTIONS = {
    "mean": "average level",
    "std": "variability",
    "min": "minimum value",
    "max": "peak value",
    "rms": "RMS level",
    "ptp": "peak-to-peak swing",
}

MEASUREMENT_NAMES = {
    "cur": "Current",
    "vol": "Voltage",
}

# Matches the 48-channel statistical feature names, e.g.
# "Bus_1_Line_01_02A_cur_L1_A_ptp" -> Bus 1, Line 01-02A, current, phase L1, peak-to-peak swing.
CHANNEL_FEATURE_PATTERN = re.compile(
    r"^Bus_(?P<bus>\d+)_Line_(?P<line_from>\d+)_(?P<line_to>\d+)(?P<segment>[A-Z])"
    r"_(?P<measure>cur|vol)_L(?P<phase>\d)_[AV]_(?P<stat>mean|std|min|max|rms|ptp)$"
)


def _describe_channel_feature(feature_name: str) -> str | None:
    match = CHANNEL_FEATURE_PATTERN.match(feature_name)
    if match is None:
        return None
    parts = match.groupdict()
    measurement = MEASUREMENT_NAMES.get(parts["measure"], parts["measure"])
    stat = STAT_DESCRIPTIONS.get(parts["stat"], parts["stat"])
    location = (
        f"Bus {parts['bus']}, Line {parts['line_from']}-{parts['line_to']}{parts['segment']}, "
        f"phase L{parts['phase']}"
    )
    return f"{measurement} {stat} on {location} contributed to this prediction"


def feature_message(feature_name: str) -> str:
    if feature_name in FEATURE_MESSAGES:
        return FEATURE_MESSAGES[feature_name]
    if feature_name in CONTEXT_MESSAGES:
        return CONTEXT_MESSAGES[feature_name]
    described = _describe_channel_feature(feature_name)
    if described is not None:
        return described
    return f"{feature_name} contributed to this prediction"
