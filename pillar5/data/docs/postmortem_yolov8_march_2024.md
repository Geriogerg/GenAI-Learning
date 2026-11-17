# Postmortem - YOLOv8 accuracy drop on line 3 (March 2024)

Author: Thomas. Status: closed. Severity: medium (no defective part shipped to customers,
but around 300 parts had to be re-inspected manually).

## Summary

In early March 2024 the yolov8 defect detector started missing small scratches on line 3.
The root cause was a change of camera supplier on line 3, which was not communicated to the
ML team. The new cameras have a different white balance and a lower resolution, so the
images no longer looked like the training data.

## Timeline

- 2024-02-26: maintenance replaces the 4 cameras of line 3 with a cheaper model from a new
  camera supplier. Resolution goes from 1920x1080 to 1280x720.
- 2024-03-04: the weekly drift job flags yolov8 with a PSI of 0.31.
- 2024-03-05: operators report missed detections on small scratches.
- 2024-03-06: root cause found by comparing images before and after 02-26.
- 2024-03-07: temporary mitigation deployed (see below).
- 2024-03-18: retraining of v2.1 on 12,400 new images from the new cameras.
- 2024-03-28: v2.1 enters the A/B test.

## Impact

On a re-labeled sample of 2,000 images from the new cameras, the mAP@0.5 of yolov8 on line 3
dropped from 0.876 to 0.812. Recall on scratches smaller than 2 mm was the most affected.

## Mitigation

As a temporary fix we lowered the confidence threshold from 0.5 to 0.35 on line 3. This
brought recall back close to the previous level, at the cost of more false positives
(about 4% of parts sent to manual inspection instead of 1.5%).

## Lessons learned

- Camera, lens and lighting changes must now go through the change-management checklist,
  with the ML team as a mandatory reviewer.
- The drift job will report PSI per production line, not only globally. Line 3 alone was
  drifting for a week before the global metric crossed the threshold.
- We should keep a small labeled set per line to measure accuracy, not only drift.
