from unittest.mock import Mock
import pytest
from PIL import Image
from app.src.util.ingest import txt_text, pdf_text, ocr_png
from app.src.util.ingest.limits import IngestionLimitExceeded
from app.src.common.config.base_config import settings


def test_txt_rejects_byte_and_text_limits(tmp_path, monkeypatch):
    path = tmp_path / 'input.txt'
    path.write_text('abcdef', encoding='utf-8')
    monkeypatch.setattr(settings, 'ingest_max_file_bytes', 5)
    with pytest.raises(IngestionLimitExceeded, match='FILE_BYTES'):
        txt_text.extract_txt(str(path))
    monkeypatch.setattr(settings, 'ingest_max_file_bytes', 10)
    monkeypatch.setattr(settings, 'ingest_max_text_chars', 5)
    with pytest.raises(IngestionLimitExceeded, match='TEXT_CHARS'):
        txt_text.extract_txt(str(path))
    path.write_text('abc', encoding='utf-8')
    assert txt_text.extract_txt(str(path)) == 'abc'


def test_pdf_checks_pages_before_extraction_and_text_before_retention(tmp_path, monkeypatch):
    path = tmp_path / 'input.pdf'
    path.write_bytes(b'fake')
    page = Mock()
    page.extract_text.return_value = 'abc'
    monkeypatch.setattr(pdf_text, 'PdfReader', Mock(return_value=Mock(pages=[page, page])))
    monkeypatch.setattr(settings, 'ingest_max_pdf_pages', 1)
    with pytest.raises(IngestionLimitExceeded, match='PDF_PAGES'):
        pdf_text.extract_pdf(str(path))
    page.extract_text.assert_not_called()
    monkeypatch.setattr(settings, 'ingest_max_pdf_pages', 2)
    monkeypatch.setattr(settings, 'ingest_max_text_chars', 5)
    with pytest.raises(IngestionLimitExceeded, match='TEXT_CHARS'):
        pdf_text.extract_pdf(str(path))


def test_png_checks_pixels_before_preprocessing_and_passes_timeout(tmp_path, monkeypatch):
    path = tmp_path / 'input.png'
    Image.new('RGB', (10, 10)).save(path)
    preprocess = Mock(wraps=ocr_png._preprocess)
    monkeypatch.setattr(ocr_png, '_preprocess', preprocess)
    model = Mock(return_value='text')
    monkeypatch.setattr(ocr_png.pytesseract, 'image_to_string', model)
    monkeypatch.setattr(settings, 'ingest_max_image_pixels', 99)
    with pytest.raises(IngestionLimitExceeded, match='IMAGE_PIXELS'):
        ocr_png.extract_png(str(path))
    preprocess.assert_not_called()
    model.assert_not_called()
    monkeypatch.setattr(settings, 'ingest_max_image_pixels', 100)
    monkeypatch.setattr(settings, 'ingest_ocr_timeout_seconds', 7)
    assert ocr_png.extract_png(str(path)) == 'text'
    assert model.call_args.kwargs['timeout'] == 7
