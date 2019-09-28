from src.models.feature_extractor import maybe_download_and_extract
from src.utils.config import get_config_from_json


if __name__ == "__main__":
    config, _ = get_config_from_json("configs/configs.json")
    maybe_download_and_extract(config.paths.model_dir, config.paths.data_url)


