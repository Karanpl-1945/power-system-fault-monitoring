FEATURE_MESSAGES = {
    "I_A_rms": "Current RMS on phase A is unusually high",
    "I_B_rms": "Current RMS on phase B is unusually high",
    "I_C_rms": "Current RMS on phase C is unusually high",
    "V_A_rms": "Voltage RMS on phase A changed significantly",
    "V_B_rms": "Voltage RMS on phase B changed significantly",
    "V_C_rms": "Voltage RMS on phase C changed significantly",
}


def feature_message(feature_name: str) -> str:
    return FEATURE_MESSAGES.get(feature_name, f"{feature_name} contributed to this prediction")

