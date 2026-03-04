from pydantic_settings import BaseSettings
from pathlib import Path
from pydantic import Field


class Settings(BaseSettings):
    qwen_api_key: str
    qwen_model: str = "qwen-turbo"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 使用“以项目代码位置为基准”的绝对路径，避免不同启动目录造成多个 data/lifeops.db
    database_url: str = f"sqlite:///{(Path(__file__).resolve().parent.parent / 'data' / 'lifeops.db').as_posix()}"

    docs_dir: str = r"E:\work\agent\docu"
    tesseract_cmd: str | None = None
    log_level: str = "INFO"

    # OCR 配置（用于 PNG 图片文字识别）
    # - OCR_LANG: tesseract 语言包，例如 "chi_sim+eng"
    # - OCR_PSM/OCR_OEM: tesseract 参数，psm 6 对“列表/块状文字”截图更友好
    # - OCR_DEBUG: 1 时保存预处理后的图片到 ./data/ocr_debug 便于肉眼检查
    ocr_lang: str = Field(default="chi_sim+eng", alias="OCR_LANG")
    ocr_psm: int = Field(default=6, alias="OCR_PSM")
    ocr_oem: int = Field(default=3, alias="OCR_OEM")
    ocr_debug: bool = Field(default=False, alias="OCR_DEBUG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
