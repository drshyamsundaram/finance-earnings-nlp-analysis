# Architecture

The project uses a lightweight layered architecture:

1. Input layer for transcript ingestion.
2. Text processing layer for normalization, sentence splitting, and tokenization.
3. Metrics layer for modular NLP analyses.
4. Export layer for CSV outputs and manifests.
5. Presentation layer for interactive HTML generation.

This keeps analyst extracts independent while allowing a single orchestration path for CLI and automation use.
