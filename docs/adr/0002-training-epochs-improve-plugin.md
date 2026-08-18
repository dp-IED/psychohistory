# Training epochs improve the plugin, not model weights

A training epoch is a scored pass whose output is skill/tool/reference changes, with an explicit guard against skill bloat. Foundation weights stay frozen. Fine-tuning from these forecasts is out of scope until that product is chosen separately.

This is easy to confuse with using historical Brier as a weight-training label. The plugin loop does not.
