
from huggingface_hub import hf_hub_download
import json

adapters = [
    "Akarshika/qwen25-3b-text2sql-lora",
    "madokalif/qwen2.5-3b-translation-warmup-adapter",
    "balaguhanesh/tool-call-ft",
    "Pilestin/qwen2.5-3b-debug-v1-lora",
]

expected_modules = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
}

for repo in adapters:
    path = hf_hub_download(
        repo_id=repo,
        filename="adapter_config.json"
    )

    with open(path) as f:
        cfg = json.load(f)

    print(f"\n{repo}")
    print("Base:", cfg.get("base_model_name_or_path"))
    print("Rank:", cfg.get("r"))
    print("Alpha:", cfg.get("lora_alpha"))
    print("Targets:", cfg.get("target_modules"))
    print("RSLoRA:", cfg.get("use_rslora", False))
    print("DoRA:", cfg.get("use_dora", False))

    assert cfg.get("r") == 16, "Unexpected rank"
    assert set(cfg.get("target_modules", [])) == expected_modules
    assert cfg.get("peft_type") == "LORA"
    assert cfg.get("use_dora", False) is False

    base = cfg.get("base_model_name_or_path", "")
    assert "Qwen2.5-3B-Instruct" in base

