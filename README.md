# MSW 造型防盜拆解工具 MSW Skin Fragmenter Pro
本工具可將含透明區的 PNG 主圖隨機分割為多個碎片，並自動生成干擾像素，有效提升美術資源的防盜還原難度。支援碎片管理、還原預覽、ZIP 匯出與多項進階干擾合成功能。


## 主要功能
- 主圖/遮罩載入：支援遮罩分割與反轉，分割參數彈性調整
- 碎片分割：可設定碎片數量、重疊像素、隨機度、聚合度
- 碎片管理：合併、複製、刪除、批次命名、排序、垃圾桶復原
- 進階干擾：一鍵生成干擾像素，支援劣化處理與陷阱圖塊（可強化碎片不可逆）
- 碎片還原預覽：即時檢視還原結果
- ZIP 匯出：碎片一鍵打包下載
- 多核心加速：大幅提升分割效能


## 適用環境
- Windows 10/11（建議 8GB 記憶體以上）
- Python 3.8+（原始碼版本）
- 提供 EXE 免安裝版

## 下載與執行
- **Windows 用戶**：  
- 至 [GitHub Release]((https://github.com/duoduo-88/MSW-Skin-Fragmenter-Pro/releases))
- 下載 EXE 檔，解壓後雙擊執行，**無需安裝 Python**。

**原始碼用戶**：  
- 安裝相依套件：
- pip install -r requirements.txt
- 執行：
- python msw_skin_fragmenter_pro_v1.1.1.py

## 注意事項
- 若圖片過大或碎片數量過多，運算時間可能較長、甚至卡頓。
- 建議：請先以小圖或較少碎片測試功能與流程。
- 工具僅供學術、技術交流及防盜研究用途，請勿用於非法行為。
- 部分 Windows Defender 可能會誤判 EXE，請自行加入信任清單。
- 本工具僅於本地瀏覽器執行，不會將圖片檔案上傳至網路。
- 防盜分割無法絕對保證圖片「不可還原」，安全性取決於主圖內容及參數配置。
- 參考密碼學 Kerckhoffs's Principle：「安全性應建立於輸入的不確定性，而非演算法本身」 

## 授權
- 本專案採用 MIT License，使用者需自負風險，作者不承擔任何法律責任。

MIT License

Copyright (c) 2025 DuoDuo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

作者：DuoDuo
發布：2025
