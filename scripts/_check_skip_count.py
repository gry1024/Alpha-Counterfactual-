from datetime import datetime, timedelta

def _entries_before(start_str, target_str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    target = datetime.strptime(target_str, "%Y-%m-%d")
    n = 0
    cur = start
    while cur < target:
        if cur.weekday() < 5:
            n += 1
        cur += timedelta(days=1)
    return n

print('weekday count 2022-01-04 -> 2023-01-03:', _entries_before("2022-01-04", "2023-01-03"))
print('weekday count 2022-01-04 -> 2023-01-04:', _entries_before("2022-01-04", "2023-01-04"))