from pen import ActivityTracker

activity_tracker = ActivityTracker(None, None)
activity_tracker.import_tracker("data/indices/PenC-.pkl")

print(activity_tracker.query("feeding", 500, 8000))