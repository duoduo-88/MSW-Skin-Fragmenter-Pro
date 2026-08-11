# MSW Skin Fragmenter Pro

[English](README.md)

MSW Skin Fragmenter Pro 是一套 Windows 桌面圖像碎片整理工具，用於本機 PNG 素材製作流程，並提供繁體中文／英文介面。

## 主要功能

- 可設定分片數量，並支援主體遮罩與次要遮罩
- 碎片顯示、複選、疊圖預覽、復原與匯出管理
- 可選用劣化與干擾像素處理
- 支援 PSD 範本匯出
- 耗時工作採背景處理並顯示進度
- 繁體中文／英文介面切換

## 下載

Windows 打包版本請至 [GitHub Releases](https://github.com/duoduo-88/MSW-Skin-Fragmenter-Pro/releases) 下載。本儲存庫與 Release 套件均不提供 PSD 範本檔案。

## 原始碼需求

- Python 3.13（已測試）
- PySide6
- NumPy
- Pillow
- psd-tools

相依套件列於 `requirements.txt`，程式入口為 `MSW Skin Fragmenter Pro v1.3.0.py`。

## 重要說明

碎片化與視覺干擾屬於混淆措施，不是加密或數位版權管理。它們可以增加一般擷取的成本，但無法保證用戶端能顯示的素材永遠無法被還原。

請只處理自己擁有或已取得授權的素材。本專案不鼓勵侵權、未授權擷取或規避第三方保護措施。

## 支持作者

如果這個專案對你有幫助，可以透過 [Ko-fi](https://ko-fi.com/duoduo88) 支持 DuoDuo。

## 授權

本專案原始碼採用 [MIT License](LICENSE)。隨附的第三方元件各自保留原授權，詳見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
