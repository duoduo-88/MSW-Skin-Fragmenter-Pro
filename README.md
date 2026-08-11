# MSW Skin Fragmenter Pro

[繁體中文](README_zh-TW.md)

MSW Skin Fragmenter Pro is a Windows desktop utility for preparing and managing fragmented PNG assets. It is designed for local asset-production workflows and includes a bilingual Traditional Chinese / English interface.

## Highlights

- Configurable image fragmentation with optional primary and secondary masks
- Fragment visibility, selection, layered preview, recovery, and export tools
- Optional degradation and interference-pixel processing
- PSD-template export support
- Background processing and progress feedback for longer operations
- Traditional Chinese / English interface

## Download

Download the packaged Windows build from [GitHub Releases](https://github.com/duoduo-88/MSW-Skin-Fragmenter-Pro/releases). PSD template files are not distributed with this repository or its release packages.

## Source requirements

- Python 3.13 (tested)
- PySide6
- NumPy
- Pillow
- psd-tools

Dependencies are listed in `requirements.txt`. The application entry point is `MSW Skin Fragmenter Pro v1.3.0.py`.

## Important notice

Fragmentation and visual interference are obfuscation measures, not encryption or digital-rights management. They can raise the cost of casual extraction, but cannot guarantee that assets displayed by a client application are unrecoverable.

Use this software only with assets you own or are authorized to process. The authors do not endorse copyright infringement, unauthorized extraction, or circumvention of third-party protections.

## License

Project source code is released under the [MIT License](LICENSE). Bundled third-party components retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
