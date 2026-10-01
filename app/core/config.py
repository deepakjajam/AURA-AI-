import secrets

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    app_name: str = 'AURA AI'
    # A fresh development-only secret keeps an omitted .env from using a
    # predictable signing key. Set a stable secret through the environment in production.
    secret_key: str = secrets.token_urlsafe(48)
    database_url: str = 'sqlite:///./aura.db'
    dev_model_id: str = 'Qwen/Qwen2.5-0.5B-Instruct'
    inference_backend: str = 'hybrid'
    aura_mode: str = 'auto'
    openai_api_key: str = ''
    aura_api_url: str = ''
    online_model_id: str = 'aura-online'

    @field_validator('secret_key', mode='before')
    @classmethod
    def use_random_secret_for_empty_example(cls, value):
        if value is None or not str(value).strip() or str(value).startswith('replace-with-'):
            return secrets.token_urlsafe(48)
        return value


settings = Settings()
