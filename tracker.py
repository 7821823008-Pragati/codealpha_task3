"""
tracker.py
----------
A simple Centroid Tracking algorithm.

Idea:
1. Every detected object has a centre point (centroid).
2. In each new frame we compare the new centroids with the centroids
   of objects from the previous frame.
3. The closest pairs are treated as the SAME object (same ID).
4. A detection with no match gets a NEW ID.
5. An object that is not found for several frames is removed.
"""

import numpy as np


def calculate_centroid(box):
    """Return the centre point (x, y) of a box given as (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)


class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=80):
        # max_disappeared : frames an object may be missing before we forget it
        # max_distance    : max pixels an object may move between two frames
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.next_id = 1          # the next new ID to hand out
        self.objects = {}         # active objects: id -> centroid, class_name, missing
        self.all_ids = {}         # every ID ever created: id -> class_name (for counting)

    def track_objects(self, detections):
        """
        Give every detection an 'id'.
        detections : list of dicts with keys 'class_name', 'confidence', 'box'
        Returns the same list, with an 'id' key added to each dict.
        """
        # Case 1: nothing detected -> every known object is "missing" one more frame
        if len(detections) == 0:
            self._mark_missing(set())
            return detections

        centroids = [calculate_centroid(d["box"]) for d in detections]

        # Step 1: measure the distance between every old object and every new detection
        pairs = []
        for obj_id, obj in self.objects.items():
            old_x, old_y = obj["centroid"]
            for index, (new_x, new_y) in enumerate(centroids):
                # only match objects of the same class (a car cannot become a person)
                if obj["class_name"] != detections[index]["class_name"]:
                    continue
                distance = float(np.hypot(new_x - old_x, new_y - old_y))
                if distance <= self.max_distance:
                    pairs.append((distance, obj_id, index))

        # Step 2: match the closest pairs first
        pairs.sort(key=lambda pair: pair[0])
        used_ids = set()
        used_detections = set()
        for distance, obj_id, index in pairs:
            if obj_id in used_ids or index in used_detections:
                continue
            self.objects[obj_id]["centroid"] = centroids[index]
            self.objects[obj_id]["missing"] = 0
            detections[index]["id"] = obj_id
            used_ids.add(obj_id)
            used_detections.add(index)

        # Step 3: old objects that found no match are missing in this frame
        self._mark_missing(used_ids)

        # Step 4: detections with no match are new objects -> new IDs
        for index, detection in enumerate(detections):
            if index in used_detections:
                continue
            new_id = self.next_id
            self.next_id += 1
            self.objects[new_id] = {
                "centroid": centroids[index],
                "class_name": detection["class_name"],
                "missing": 0,
            }
            self.all_ids[new_id] = detection["class_name"]
            detection["id"] = new_id

        return detections

    def _mark_missing(self, matched_ids):
        """Increase the 'missing' counter of unmatched objects; delete old ones."""
        for obj_id in list(self.objects.keys()):
            if obj_id in matched_ids:
                continue
            self.objects[obj_id]["missing"] += 1
            if self.objects[obj_id]["missing"] > self.max_disappeared:
                del self.objects[obj_id]

    def get_class_counts(self):
        """Number of UNIQUE objects (unique IDs) for every class name."""
        counts = {}
        for class_name in self.all_ids.values():
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts
