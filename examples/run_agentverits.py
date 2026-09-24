from agentverits import AgentVeriTSConfig, AgentVeriTSPipeline
from agentverits.io import load_signal_csv

values, _ = load_signal_csv("data/example.csv")
config = AgentVeriTSConfig.from_yaml("configs/agentverits_default.yaml")
result = AgentVeriTSPipeline(config).run(values, signal_id="example", output_dir="outputs/example")
print(result.as_dict())
