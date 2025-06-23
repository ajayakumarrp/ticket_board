from datetime import datetime, timezone

def convert_to_timestamp(date_str):
    """
    Convert a date string (with or without time) to a Unix timestamp.
    Supports:
    - "2025-06-23"
    - "2025-06-23 14:30"
    - "2025-06-23T14:30:00"
    - "2024-12-25T00:00:00Z" (UTC)
    """
    if date_str.endswith('Z'):
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
            dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            pass

    date_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return str(dt.timestamp())  # assumes local time
        except ValueError:
            continue
    
    raise ValueError(f"Unsupported date format: '{date_str}'")

