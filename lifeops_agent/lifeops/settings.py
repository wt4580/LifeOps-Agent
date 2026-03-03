from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qwen_api_key: str
    qwen_model: str = "qwen-turbo"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    database_url: str = "sqlite:///./data/lifeops.db"
    docs_dir: str = r"E:\work\agent\docu"
    tesseract_cmd: str | None = None
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

