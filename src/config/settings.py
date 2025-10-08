"""Settings loader for LLM and Fetcher configuration"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Load and manage LLM provider configuration"""

    def __init__(
        self,
        llm_config_path: str = "src/config/llm_config.yaml",
        fetcher_config_path: str = "src/config/fetcher_config.yaml"
    ):
        # Get project root from environment or auto-detect
        project_root = os.getenv("PROJECT_ROOT")
        if project_root:
            self.project_root = Path(project_root)
        else:
            # Auto-detect: go up from this file to project root
            self.project_root = Path(__file__).parent.parent.parent

        # Build absolute paths
        self.llm_config_path = self.project_root / llm_config_path
        self.fetcher_config_path = self.project_root / fetcher_config_path

        self._config = self._load_config(self.llm_config_path)
        self._fetcher_config = self._load_config(self.fetcher_config_path) if self.fetcher_config_path.exists() else {}
        self._validate_config()

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration"""
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _validate_config(self):
        """Validate configuration"""
        active = self.active_provider
        if active not in self._config:
            raise ValueError(f"Provider '{active}' not found in config")

        # Check API keys for OpenAI
        if active == "openai":
            api_key_env = self._config["openai"]["api_key_env"]
            if not os.getenv(api_key_env):
                print(f"Warning: Environment variable {api_key_env} not set")

    @property
    def active_provider(self) -> str:
        """Get active provider"""
        return self._config["active_provider"]

    def get_provider_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get provider configuration"""
        provider = provider or self.active_provider
        return self._config.get(provider, {})

    def get_api_key(self, provider: Optional[str] = None) -> str:
        """Get provider API key"""
        provider = provider or self.active_provider
        config = self.get_provider_config(provider)

        if "api_key_env" in config:
            env_var = config["api_key_env"]
            return os.getenv(env_var, "")

        return ""  # Ollama doesn't require API key

    def get_model(self, model_type: str, provider: Optional[str] = None) -> str:
        """Get model name"""
        provider = provider or self.active_provider
        config = self.get_provider_config(provider)
        return config["models"].get(model_type, "")

    def get_base_url(self, provider: Optional[str] = None) -> str:
        """Get provider base URL"""
        provider = provider or self.active_provider
        config = self.get_provider_config(provider)
        return config.get("base_url", "")

    def get_general_config(self) -> Dict[str, Any]:
        """Get general settings"""
        return self._config.get("general", {})

    # ==================== Fetcher methods ====================

    @property
    def active_fetcher(self) -> str:
        """Get active fetcher"""
        return self._fetcher_config.get("active_fetcher", "pubmed")

    def get_fetcher_config(self, fetcher: Optional[str] = None) -> Dict[str, Any]:
        """Get fetcher configuration"""
        fetcher = fetcher or self.active_fetcher
        return self._fetcher_config.get(fetcher, {})

    def get_fetcher_api_key(self, fetcher: Optional[str] = None) -> str:
        """Get fetcher API key"""
        fetcher = fetcher or self.active_fetcher
        config = self.get_fetcher_config(fetcher)

        if "api_key_env" in config:
            env_var = config["api_key_env"]
            return os.getenv(env_var, "")

        return ""

    def get_fetcher_base_url(self, fetcher: Optional[str] = None) -> str:
        """Get fetcher base URL"""
        fetcher = fetcher or self.active_fetcher
        config = self.get_fetcher_config(fetcher)
        return config.get("base_url", "")


# Global instance
settings = Settings()
