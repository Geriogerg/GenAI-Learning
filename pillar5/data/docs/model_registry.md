# CV Model Registry

Last updated: 2024-04-02 by Thomas (ML Platform)

This page lists every computer vision model we track, where it runs and who owns it.
If you deploy something new, add it here before the release, not after.

## Summary table

| Model | Task | Dataset | Accuracy | Latency p95 (Orin) | Params | Owner | Status |
|---|---|---|---|---|---|---|---|
| resnet50 | classification | imagenet | 92.3% | 45 ms | 25M | Marie | production |
| yolov8 | defect detection | line3-defects | mAP@0.5 0.876 | 12 ms | 3M | Thomas | production, v2.1 in A/B test |
| efficientnet | classification | imagenet | 90.1% | 30 ms | 8M | Julien | production |
| vit_large | classification | imagenet | 95.1% | 120 ms | 307M | Marie | not deployed |

## resnet50

Our oldest model, used to classify parts into 14 product families at the entrance of the
lines. Owned by Marie. It has been stable for more than a year and drift monitoring has never
flagged it. Last trained on 2024-01-15. Nobody is planning to replace it this year, the
accuracy is good enough and the latency budget is comfortable.

## yolov8

The defect detector running on line 3 (scratches, dents, missing screws). Owned by Thomas.
This is the model that caused the March 2024 incident, see the postmortem page for details.
Version v2.1, retrained on 12,400 images from the new cameras, has been in A/B test since
2024-03-28. Until the A/B test is finished, v2.0 still serves 50% of the traffic.

## efficientnet

EfficientNet-B3 is used for the surface finish classification (matte / glossy / defective
coating). It is owned by Julien, who joined the team in January. It was retrained on
2024-02-10 after a slow performance degradation. Drift monitoring flagged it again in
March (PSI 0.14, warning level only) and Julien is watching it.

## vit_large

ViT Large has the best accuracy we have measured (95.1% on our imagenet benchmark) but it is
not deployed. Its latency is 120 ms on the Orin, way above the 50 ms budget from the
deployment runbook. Two options were discussed: distillation into a smaller ViT, or
quantization to INT8. The team decided to deprioritize it until someone can show a version
under 50 ms. Owner: Marie.
