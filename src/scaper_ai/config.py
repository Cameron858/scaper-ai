from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    chat_model: str = "llama3.2"
    embed_model: str = "nomic-embed-text"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()
