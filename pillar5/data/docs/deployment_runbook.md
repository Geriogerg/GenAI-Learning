# Model deployment runbook

How to get a CV model from a notebook to the production lines. Follow the steps in order.

## Step 1 - Model card

Open a pull request on the model-cards repository describing the model: training data,
metrics, known limitations, owner. No model card, no deployment.

## Step 2 - Offline evaluation

The new model must beat the current production model by at least 0.5 point on the frozen
test set golden-v3. The test set is never used for training or for picking checkpoints.
If you need a new test set, talk to the platform team first.

## Step 3 - Latency check

Run the benchmark script on the Jetson Orin test bench. The p95 latency must be below 50 ms
per image, including pre-processing. Models above the budget are rejected even if they are
more accurate (see vit_large in the registry).

## Step 4 - Canary

Deploy to 5% of the traffic for 24 hours. The canary is rolled back automatically if the
error rate goes above 2% or if the p95 latency goes above 60 ms in production.

## Step 5 - A/B test

Run an A/B test with 50% of the traffic on each version for 48 hours. We compare the defect
recall measured on the operators' manual checks, not only the model confidence.

## Step 6 - Approval and rollout

Full rollout needs two approvals: Sophie (product) and Thomas (platform). Then deploy to
100% with the release pipeline.

## Rollback

To roll back a model, run `mlctl rollback <model> --to <version>`. It takes less than two
minutes and does not need any approval. Always post a message in #cv-monitoring afterwards.

## Deployment freeze

No deployments on Fridays after 14:00, and no deployments during the last week of December.
Emergency rollbacks are allowed at any time.
