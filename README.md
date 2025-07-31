# MSW 造型防盜拆解工具

本工具可將含透明區的 PNG 主圖隨機分割為多個碎片，並能產生干擾像素，提升美術資源防盜還原難度。支援碎片管理、還原預覽、ZIP 匯出及進階干擾合成。

## 主要功能

- 載入主圖與遮罩，自訂分割參數，一鍵產生碎片
- 支援重疊像素、碎片隨機度、聚合度等進階調整
- 可產生干擾像素並一鍵套用到碎片
- 碎片合併、複製、刪除、批次命名、排序
- 支援碎片 ZIP 匯出及垃圾桶復原
- 支援多核心加速

## 適用環境

- Windows 10/11（建議 8GB 記憶體以上）
- Python 3.8+（僅限原始碼版本）
- EXE 免安裝直接執行

## 下載與執行

- Windows 用戶可直接到 ([GitHub Release 區](https://github.com/duoduo-88/MSW-Skin-Fragmenter/releases/tag/v1.0.0](https://github.com/duoduo-88/MSW-Skin-Fragmenter/releases))) 下載 exe 檔，解壓後雙擊執行，無需安裝 Python。
- 原始碼用戶請安裝 `requirements.txt`，然後執行 `python msw_skin_fragmenter.py`。

## 注意事項

- 圖片檔案過大或碎片數過多時，執行可能耗時或卡頓。
- 強烈建議優先測試較小圖片與合理碎片數。
- 本工具僅供學術、技術交流與防盜研究用途，請勿用於非法行為。  
- EXE 版本若被 Windows Defender 誤判，請自行加入信任。

## 授權

MIT License。使用者需自負風險，作者不負任何責任。

作者：DuoDuo  
發布：2025
