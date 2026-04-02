"""Pytest configuration and fixtures.

Sets required environment variables before any module imports so that
LabConfig() (which runs at module import time) does not fail during testing.
Real API calls are mocked at the test level; these values are never sent.
"""
import os

# Must be set before rag_quality_lab is imported
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("DATABASE_URL", "postgresql://lab:lab@localhost:5432/raglab")
