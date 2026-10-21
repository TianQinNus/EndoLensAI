# E1 Mac validation scope

Historical CPU synthetic checks covered paired augmentation, per-image loss/gradient accumulation, probability restoration and metrics, checkpoint recovery and uninterrupted-versus-resumed small-model equality. A standard31,031,745-parameter352 CPU smoke was recorded during preparation.

CPU tests do not validate MPS drivers, current installation compatibility, full GPU training or cross-backend bitwise equality. The actual later Windows/Mac experiment is documented in the [baseline report](../../../ProjectReport/E0_E4_BASELINES.md). No new MPS training was run for this English migration.
