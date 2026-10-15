from collections import Counter

import cv2
import numpy as np
from sklearn.cluster import KMeans


def shirt_feature(frame, box):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    crop = frame[max(0, int(top + height * 0.18)):int(top + height * 0.50), max(0, int(left + width * 0.25)):int(left + width * 0.75)]
    if crop.size < 30:
        return None
    pixels = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    hue, saturation, value = np.median(pixels, axis=0)
    angle = hue / 180 * 2 * np.pi
    saturation /= 255
    return np.array([np.cos(angle) * saturation, np.sin(angle) * saturation, saturation, value / 255 * 0.25])


class TeamClassifier:
    def __init__(self):
        self.model = None
        self.votes = {}

    def fit(self, features):
        if len(features) >= 4:
            self.model = KMeans(n_clusters=2, n_init=10, random_state=0).fit(np.asarray(features))

    def predict(self, frame, box, track_id):
        feature = shirt_feature(frame, box)
        if self.model is None or feature is None:
            return "unknown"
        cluster = int(self.model.predict(feature.reshape(1, -1))[0])
        self.votes.setdefault(track_id, Counter())[cluster] += 1
        winner = self.votes[track_id].most_common(1)[0][0]
        return f"cluster_{winner}"
