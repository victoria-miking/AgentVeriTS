from reva import REVAConfig, REVAPipeline
from reva.io import load_signal_csv

values, _ = load_signal_csv("data/example.csv")
config = REVAConfig.from_yaml("configs/reva_default.yaml")
result = REVAPipeline(config).run(values, signal_id="example", output_dir="outputs/example")
print(result.as_dict())
