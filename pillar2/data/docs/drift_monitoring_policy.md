# Drift monitoring policy

Owner: ML Platform team. Applies to every model with status "production" in the registry.

## Why we monitor drift

Our models are trained on images from a specific setup: cameras, lighting, product mix.
When any of these change, the input distribution moves and accuracy drops silently, because
we don't get labels in real time. Drift monitoring is our early warning system.

## The weekly drift job

The drift job runs every Monday at 06:00 UTC on the images of the previous 7 days.
For each production model it computes two checks:

1. Input drift: we take the embeddings of the penultimate layer, project them to 16
   dimensions with a PCA fitted on the training set, and compute the Population Stability
   Index (PSI) on each dimension. We report the maximum PSI over the 16 dimensions.
2. Prediction drift: we compare the distribution of predicted classes with the training
   distribution using a chi-square test.

## Thresholds

- PSI below 0.1: stable, nothing to do.
- PSI between 0.1 and 0.25: warning. The owner is notified but no action is required.
- PSI above 0.25: drift detected.
- Prediction drift is flagged when the chi-square p-value is below 0.01.

## What happens when drift is detected

An alert is posted in the #cv-monitoring Slack channel and a ticket is opened on the ML-OPS
Jira board, assigned to the model owner. The owner has 5 business days to triage: confirm
the drift, find the root cause, and decide if retraining is needed.

## Retraining rules

Retraining is triggered automatically only when at least 10,000 new labeled samples are
available for the model. Labeling is done by our vendor LabelCo with a 3-day turnaround.
Below 10,000 samples, retraining is a manual decision of the model owner.

To keep GPU costs under control, a model can be retrained at most 2 times per quarter
without approval from the team lead. Every retrained model goes through the full
deployment runbook, there is no shortcut for "just a retrain".
