from datetime import time


DEFAULT_SHIFT_NAME = 'Ca hành chính'
DEFAULT_SHIFT_START = time(8, 0)
DEFAULT_SHIFT_END = time(17, 0)
DEFAULT_SHIFT_WORK_DAYS = [0, 1, 2, 3, 4]
DEFAULT_LATE_THRESHOLD = 15


def get_or_create_default_shift():
    from .models import Shift

    shift, _ = Shift.objects.update_or_create(
        name=DEFAULT_SHIFT_NAME,
        defaults={
            'start_time': DEFAULT_SHIFT_START,
            'end_time': DEFAULT_SHIFT_END,
            'work_days': DEFAULT_SHIFT_WORK_DAYS,
            'late_threshold': DEFAULT_LATE_THRESHOLD,
        },
    )
    return shift
