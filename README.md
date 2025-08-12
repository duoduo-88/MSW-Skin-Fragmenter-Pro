# MSW造型防盜拆解工具 專業版 | MSW Skin Fragmenter Pro

  本工具可將含透明區的 PNG 主圖隨機分割為多個碎片，並自動生成干擾像素，有效提升美術資源的防盜還原難度。  
  This tool can randomly split a PNG image with transparent areas into multiple fragments and automatically generate interference pixels, effectively increasing the difficulty of restoring stolen artwork.
  
  支援碎片管理、還原預覽、ZIP 匯出與多項進階干擾合成功能。  
  Supports fragment management, restoration preview, ZIP export, and multiple advanced interference blending features.

---

## 主要功能 | Key Features

- **主圖/遮罩載入** | **Main Image / Mask Loading**  
  支援遮罩分割與反轉，分割參數彈性調整  
  Supports mask-based segmentation and inversion with flexible parameter adjustment  

- **碎片分割** | **Fragment Splitting**  
  可設定碎片數量、重疊像素、隨機度、聚合度  
  Adjustable number of fragments, overlapping pixels, randomness, and clustering  

- **碎片管理** | **Fragment Management**  
  合併、複製、刪除、批次命名、排序、垃圾桶復原  
  Merge, copy, delete, batch rename, sort, and restore from trash bin  

- **進階干擾** | **Advanced Interference**  
  一鍵生成干擾像素，支援劣化處理與陷阱圖塊（可強化碎片不可逆）  
  One-click generation of interference pixels, supports degradation and trap blocks (can enhance irreversibility)  

- **碎片還原預覽** | **Restoration Preview**  
  即時檢視還原結果  
  Instantly preview restoration results  

- **ZIP 匯出** | **ZIP Export**  
  碎片一鍵打包下載  
  One-click export of all fragments  

- **多核心加速** | **Multi-Core Acceleration**  
  大幅提升分割效能  
  Significantly boosts splitting performance  

---

## 適用環境 | System Requirements

- Windows 10/11（建議 8GB 記憶體以上）  
  Windows 10/11 (8GB RAM or more recommended)  
- Python 3.8+（原始碼版本）  
  Python 3.8+ (for source code version)  
- 提供 EXE 免安裝版  
  Portable EXE version available  

---

## 下載與執行 | Download & Run

  ### Windows 用戶 | Windows Users
  - 至 [GitHub Release](https://github.com/duoduo-88/MSW-Skin-Fragmenter-Pro/releases) 下載 EXE 檔，解壓後雙擊執行  
    Download the EXE file from [GitHub Release](https://github.com/duoduo-88/MSW-Skin-Fragmenter-Pro/releases), extract it, and double-click to run  
  - **無需安裝 Python**  
    **No Python installation required**
  
  ### 原始碼用戶 | Source Code Users
  
    # 安裝相依套件 | Install dependencies
    pip install -r requirements.txt
    
    # 執行程式 | Run the program
    python msw_skin_fragmenter_pro_v1.1.1.py

---

## 注意事項 | Notes

  干擾像素與重疊像素功能非常重要，請盡可能優先使用。不要為了拆更多層碎片而犧牲這些功能的效果。如果缺乏干擾像素，現代的圖像偵測 AI 可以輕鬆還原碎片，降低防盜效果。  
  The functions of interference pixels and overlapping pixels are very important and should be prioritized whenever possible.Do not sacrifice these features just to split the image into more fragments.Without interference pixels, modern image detection AI can easily restore the fragments, reducing anti-theft effectiveness.
  
  若圖片過大或碎片數量過多，運算時間可能較長、甚至卡頓  
  Large images or too many fragments may cause longer processing times or lag
  
  建議：請先以小圖或較少碎片測試功能與流程  
  Recommendation: Test with smaller images or fewer fragments first
  
  工具僅供學術、技術交流及防盜研究用途，請勿用於非法行為  
  This tool is for academic, technical exchange, and anti-piracy research purposes only
  
  部分 Windows Defender 可能會誤判 EXE，請自行加入信任清單  
  Some Windows Defender versions may flag the EXE as suspicious; please add it to your trusted list
  
  本工具僅於本地端執行，不會將圖片檔案上傳至網路  
  This tool runs locally and does not upload image files to the internet
  
  防盜分割無法絕對保證圖片「不可還原」，安全性取決於主圖內容及參數配置  
  Anti-piracy splitting cannot absolutely guarantee that the image is unrecoverable; security depends on image content and parameter configuration
  
  參考密碼學 Kerckhoffs's Principle：「安全性應建立於輸入的不確定性，而非演算法本身」  
  Based on Kerckhoffs's Principle in cryptography: "Security should depend on uncertainty of the input, not the secrecy of the algorithm"

---
    
## 授權 | License

  本專案採用 MIT License，使用者需自負風險，作者不承擔任何法律責任。
  MIT 授權條款 著作權所有 (c) 2025 DuoDuo
  特此免費授權，允許任何取得本軟體與相關文件檔案（以下稱「軟體」）的人員，不受限制地處理本軟體，包括但不限於使用、複製、修改、合併、出版、發行、再授權及/或銷售本軟體副本，並允許本軟體提供者在符合以下條件的情況下也可這樣做：上述著作權聲明與本授權聲明應包含於本軟體的所有副本或主要部分中。
  本軟體是「按現狀」提供，不包含任何明示或暗示的保證，包括但不限於對適銷性、特定用途適用性及非侵權的保證。在任何情況下，作者或版權持有人不對因本軟體或本軟體的使用或其他交易而引起的任何索賠、損害或其他責任承擔責任，無論是在合約訴訟、侵權行為或其他方面。
  This project is licensed under the MIT License. Use at your own risk; the author assumes no legal responsibility.
  
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
