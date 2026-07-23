# MSW Skin Fragmenter Pro

[繁體中文](README_zh-TW.md)

MSW Skin Fragmenter Pro is a Windows desktop tool for dividing transparent PNG artwork into independently managed fragments. It supports mask-restricted splitting, overlap and interference generation, partial re-splitting, degradation effects, restoration previews, and PNG or ZIP export.

Current release: **v1.2.0**

[Download the Windows package](https://github.com/duoduo-88/MSW-Skin-Fragmenter-Pro/releases/latest)

> Fragmentation and interference are obfuscation aids, not encryption. They cannot guarantee that artwork is impossible to reconstruct.

## Features

- Splits a transparent PNG into 1–10 fragments with adjustable block size and size randomness.
- Supports an optional same-size PNG mask, mask inversion, and clipping fragments to the mask.
- Adds adjustable overlap pixels and controls their aggregation.
- Re-splits only a selected rectangular region while preserving the remainder.
- Reorders, renames, copies, merges, imports, exports, and deletes fragments.
- Keeps up to 99 deleted fragments in a recoverable trash bin.
- Generates interference images with density, alpha, overlap, and semi-transparent-region controls.
- Applies block degradation, noise, brightness variation, and color variation to imported images.
- Provides combined restoration and overlap previews before export.
- Uses worker threads and multiple processes for computationally intensive operations.
- Processes images locally and does not contain a network-upload feature.

## Requirements

### Windows package

- Windows 10 or 11
- A display large enough for the application's 1280 × 800 minimum window
- Sufficient memory for full-resolution RGBA images and multiple fragments

The release package includes the executable and does not require a separate Python installation.

### Python source

- Python 3.10 or newer is recommended
- Packages listed in `requirements.txt`: PySide6, NumPy, and Pillow

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the current source:

```powershell
python "MSW Skin Fragmenter Pro v1.2.0.py"
```

The repository also keeps `MSW Skin Fragmenter Pro v1.1.7.py` as an older fallback version.

## Basic Workflow

1. Select a transparent PNG as the source image.
2. Optionally load a same-size PNG mask and choose whether fragments may extend outside it.
3. Set the fragment count, block size, randomness, overlap percentage, and overlap aggregation.
4. Select **Run Split** (`執行拆解`).
5. Review, reorder, rename, or manage the generated fragments.
6. Optionally generate interference pixels or apply degradation effects.
7. Use the restoration/overlap preview to inspect the combined result.
8. Export selected fragments as PNG files or export all fragments as a ZIP archive.

For partial re-splitting, right-drag a region in the preview and then select **Partial Split** (`局部分割`).

## Input and Output

- Main image: PNG with transparency recommended
- Optional mask: PNG with the same dimensions as the main image
- Imported fragments or degradation source: image files accepted by Pillow through the application dialogs
- Fragment output: transparent PNG
- Batch output: ZIP containing PNG fragments

## Privacy

The source code reads and writes local files only. It does not include an image-upload or telemetry client. Operating-system dialogs and third-party runtime components remain subject to their own behavior and policies.

## Limitations

- One source image is processed at a time.
- Very large images, high fragment counts, small blocks, and dense interference can require significant time and memory.
- Closing the application during background work may cancel incomplete operations.
- Fragmentation effectiveness depends on the artwork and chosen parameters; no configuration guarantees irreversible output.
- Only the Windows release package is officially documented here. Other platforms may run the Python source but are not verified by this project.

## License

The project source code is licensed under the [MIT License](LICENSE). PySide6, NumPy, Pillow, and components included in packaged builds retain their own licenses; see [Third-Party Notices](THIRD_PARTY_NOTICES.md).

Copyright (c) 2025 DuoDuo
