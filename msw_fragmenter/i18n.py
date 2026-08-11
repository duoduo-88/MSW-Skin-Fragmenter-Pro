"""Runtime Traditional-Chinese/English UI translation support."""

import re

from PySide6 import QtCore, QtGui, QtWidgets


LANG_ZH_TW = "zh_TW"
LANG_EN = "en"
_language = LANG_ZH_TW


EXACT_EN = {
    # Window, preview toolbar and tabs.
    "MSW造型防盜拆解工具 專業版 MSW Skin Fragmenter Pro v": "MSW Skin Fragmenter Pro v",
    "深灰": "Dark",
    "白": "W",
    "50%灰": "50%",
    "透明網格": "Grid",
    "預覽背景：深灰": "Preview background: dark gray",
    "預覽背景：白": "Preview background: white",
    "預覽背景：50%灰": "Preview background: 50% gray",
    "預覽背景：透明網格": "Preview background: transparency grid",
    "預覽背景類型。僅影響預覽，不影響輸出。":
        "Preview background type. This affects only the preview, not exported files.",
    "重疊預覽": "Overlap",
    "當前無預覽圖": "No Preview",
    "碎片管理": "Fragments",
    "干擾像素": "Interference",
    "劣化處理": "Degrade",
    "垃圾桶": "Trash",
    "關於": "About",
    # Main controls (short English labels preserve the existing layout).
    "選擇主圖": "Load Image",
    "主圖：": "Image:",
    "載入主體遮罩": "Load Primary",
    "主體遮罩：": "Primary:",
    "載入次要遮罩": "Load Secondary",
    "載入": "Load",
    "次要遮罩：": "Secondary:",
    "移除": "Remove",
    "反轉": "Invert",
    "溢出": "Overflow",
    "不溢出": "No Overflow",
    "分片數量(1~10)：": "Count (1–10):",
    "方塊尺寸(1~30)：": "Block (1–30):",
    "尺寸隨機度(1~100)：": "Random (1–100):",
    "重疊像素比(0~100%)：": "Overlap (0–100%):",
    "重疊像素聚合(1~10)：": "Cluster (1–10):",
    "分片數量：": "Count:",
    "重疊像素比：": "Overlap:",
    "重疊像素聚合：": "Cluster:",
    "執行拆解": "Split",
    "局部分割": "Split Area",
    "還原初始分割": "Restore",
    "↑ 上移": "↑ Up",
    "↓ 下移": "↓ Down",
    "上移": "Move Up",
    "下移": "Move Down",
    "重新命名": "Rename",
    "全部匯出": "Export",
    "合併碎片": "Merge",
    "複製碎片": "Duplicate",
    "刪除碎片": "Delete",
    "匯入碎片": "Import",
    "匯出碎片": "Export",
    "進階管理 / 還原預覽": "Advanced / Preview",
    "結束進階管理": "Exit Advanced",
    "顯示選取碎片": "Show Selected",
    "隱藏選取碎片": "Hide Selected",
    "合併選取碎片": "Merge Selected",
    "複製選取碎片": "Duplicate Selected",
    "重新命名選取碎片": "Rename Selected",
    "刪除選取碎片": "Delete Selected",
    "匯出選取碎片": "Export Selected",
    "匯出選取碎片為 .psd": "Export Selected as .psd",
    "匯出此碎片": "Export Fragment",
    "全部匯出 ZIP": "Export All ZIP",
    "匯出選擇碎片": "Export Selected",
    "匯出全部碎片": "Export All",
    "匯出 .psd": "Export .psd",
    "方案：": "Plan:",
    "儲存方案": "Save Plan",
    "刪除方案": "Delete Plan",
    "方案名稱：": "Plan name:",
    "覆蓋方案": "Overwrite Plan",
    # Interference panel.
    "干擾像素尺寸(1~30)：": "Block Size (1–30):",
    "干擾像素尺寸：": "Block:",
    "干擾密度：5%": "Density: 5%",
    "取樣不透明度下限：1%": "Sample Alpha Min: 1%",
    "取樣不透明度上限：100%": "Sample Alpha Max: 100%",
    "允許干擾像素重疊": "Allow Block Overlap",
    "第 3 片起隨機使用前方碎片範圍": "Random Prior Scope (3+)",
    "忽略半透明區域": "Ignore Semi-Transparency",
    # Degrade panel.
    "匯入來源圖": "Load Source",
    "尚未載入任何圖": "No image loaded",
    "目前來源：": "Src:",
    "方塊尺寸：": "Block:",
    "尺寸隨機度：": "Rand:",
    "劣化密度：70%": "Density: 70%",
    "噪點強度：10%": "Noise: 10%",
    "隨機明暗：10%": "Brightness: 10%",
    "色偏強度：10%": "Color Shift: 10%",
    "產生劣化預覽": "Preview",
    "還原原圖": "Orig.",
    "掛載干擾像素": "Mount",
    "已掛載干擾像素": "Mounted",
    "製作劣化碎片": "Make Degraded Fragments",
    "製作劣化碎片進度": "Degraded Fragment Progress",
    "缺少劣化來源": "Degradation Source Required",
    "劣化處理仍在執行中": "Degradation is still running",
    "正在依目前參數處理劣化來源圖...": "Applying current degradation settings...",
    "製作劣化碎片：正在套用目前劣化參數...": "Degraded fragments: applying current settings...",
    # Trash and About.
    "垃圾桶 (可復原, 最多99項)": "Trash (restorable, max 99)",
    "復原選擇碎片": "Restore Selected",
    "清空垃圾桶": "Empty Trash",
    "圖片碎片拆解、干擾像素與劣化處理工具": "Fragmentation, interference and degradation tool",
    "用途與風險": "Purpose and Risk",
    "版本與授權": "Version and License",
    "版本：": "Version:",
    "作者：": "Author:",
    "支持作者：": "Support:",
    "開源授權：": "License:",
    "著作權：": "Copyright:",
    # PSD dialog.
    "PSD 匯出設定": "PSD Export Settings",
    "匯出 .psd 設定": "Export .psd",
    "PSD 範本資料夾：": "PSD Template Folder:",
    "開啟資料夾": "Open Folder",
    "重新整理": "Refresh",
    "碎片": "Fragment",
    "PSD 檔案": "PSD File",
    "PSD 圖層": "PSD Layer",
    "匯出檔名前綴：": "Filename Prefix:",
    "例如：角色_（可留空）": "Example: Character_ (optional)",
    "＋ 原始 PSD 檔名": "+ original PSD filename",
    "匯出": "Export",
    "取消": "Cancel",
    "是": "Yes",
    "否": "No",
    "確定": "OK",
    "（PSD 沒有可取代的圖層）": "(PSD has no replaceable layers)",
    "碎片": "Fragment",
    "主圖": "Source Image",
    "掛載劣化圖": "Mounted Degraded Image",
    "合併碎片_": "Merged_",
    "_複製": "_Copy",
    "_復原": "_Restored",
    "_干擾": "_Interference",
    "所有碎片.zip": "All Fragments.zip",
    "儲存碎片": "Save Fragment",
    "儲存所有碎片": "Save All Fragments",
    "載入主體遮罩": "Load Primary",
    "載入次要遮罩": "Load Secondary",
    # Common dialogs and statuses.
    "錯誤": "Error",
    "警告": "Warning",
    "缺少主圖": "Image Required",
    "缺少主體遮罩": "Primary Mask Required",
    "未框選區域": "No Area Selected",
    "遮罩無法使用": "Invalid Mask",
    "局部分割失敗": "Area Split Failed",
    "匯出失敗": "Export Failed",
    "請選擇碎片": "Select Fragments",
    "重新命名": "Rename",
    "名稱重複": "Duplicate Name",
    "找不到碎片": "Fragment Not Found",
    "無預覽": "No Preview",
    "產生完成": "Generation Complete",
    "尚未產生": "Nothing Generated",
    "合成完成": "Composite Complete",
    "還原初始設定": "Restore Initial State",
    "PSD 匯出進度": "PSD Export Progress",
    "PSD 匯出完成": "PSD Export Complete",
    "PSD 匯出失敗": "PSD Export Failed",
    "PSD 正在匯出": "PSD Export Running",
    "背景處理正在停止": "Stopping Background Work",
    "批次命名": "Batch Rename",
    "輸入新名稱": "Enter a new name",
    "下次重新啟動前不再顯示": "Do not show again until restart",
    "PNG圖檔 (*.png)": "PNG Images (*.png)",
    "ZIP 壓縮檔 (*.zip)": "ZIP Archives (*.zip)",
    # Corrected explanatory copy.
    "本工具僅供技術交流與學術用途，不保證碎片不可被還原。\n\n使用者需自行評估並承擔使用本工具所產生的所有風險。":
        "This tool is provided for technical exchange and academic use. It does not guarantee that fragments cannot be reconstructed.\n\nUsers must assess and accept all risks arising from its use.",
    "以上參數會在「執行拆解」與局部分割時自動套用；清單前兩張保持原樣，干擾由清單第三張開始。":
        "These settings are applied automatically during full and area splitting. The first two list items remain unchanged; interference starts from the third.",
    "為每個碎片選擇 PSD 範本與要取代的圖層。相同 PSD 範本的碎片會合併到同一份輸出檔。":
        "Choose a PSD template and target layer for each fragment. Fragments using the same template are merged into one output file.",
    "選到同一範本與同一圖層的多個碎片，會依目前碎片清單順序疊合後寫入該圖層。":
        "Fragments assigned to the same template and layer are composited in the current list order before being written.",
    "依 Photoshop 面板由上到下列出圖層，索引路徑仍指向原始結構。":
        "Layers are listed top-to-bottom as in Photoshop while their index paths continue to reference the original structure.",
    "按範本分組匯出；同一範本的多個碎片只產生一份 PSD。":
        "Exports are grouped by template; multiple fragments using one template produce one PSD.",
    "請先把 .psd 檔案放進上方的 PSD 範本資料夾，再按「重新整理」。":
        "Place .psd files in the template folder above, then select Refresh.",
}


# Long help text is kept out of compact buttons and translated here for the ? tips.
EXACT_EN.update({
    "切換繁體中文／English；不改變目前介面尺寸":
        "Switch Traditional Chinese / English without changing the UI size.",
    "請上傳含有透明區的 PNG 檔案作為主圖進行切割。透明像素將不會參與分割。":
        "Load a PNG with transparency as the source image. Transparent pixels are excluded from splitting.",
    "先在左側預覽用『右鍵拖曳』框選區域，再按此鍵只重分割該區域":
        "Right-drag in the left preview to select an area, then use this button to split only that area.",
    "遮罩流程會依已載入的遮罩執行：只載入主體遮罩時，第一張為主體外框，主體內部拆成後續碎片；只載入次要遮罩時，第一張為次要遮罩內部，次要外框拆成後續碎片；兩張都有時，依「主要內／外 → 次要內／外」順序處理。\n\n主體「溢出」只讓主體外框在主要內／外分離時向內部延伸，預設關閉。兩種溢出都不會延後到最終碎片拆分。\n\n遮罩必須是 PNG，且大小與主圖完全一致；alpha 大於 0 的像素視為遮罩範圍。\n\n若兩張遮罩都不載入，則執行基本拆分。最終名稱仍為碎片 1～N，共 N 張。":
        "The mask workflow follows the loaded masks. With only a primary mask, the first item is the primary frame and the primary interior is split into later fragments. With only a secondary mask, the first item is the secondary interior and the secondary frame is split into later fragments. With both masks, processing follows Primary Inner/Outer → Secondary Inner/Outer.\n\nPrimary Overflow lets only the primary frame extend into the interior during primary separation and is disabled by default. Neither overflow is deferred to final fragment splitting.\n\nMasks must be PNG files matching the source dimensions. Pixels with alpha above 0 are inside the mask.\n\nWith no masks, a basic split is performed. Final names remain Fragment 1–N, for N total items.",
    "次要遮罩可單獨使用；有主體遮罩時只處理主體內部，沒有主體遮罩時直接處理主圖。次要內部會保留到第一張；次要外框拆成後續碎片。勾選「溢出」時，次要外框只在這次內／外分離時溢進次要內部；預設開啟，並使用方塊尺寸與隨機度各一半的參數（最低為 1）。關閉後會嚴格限制在次要外框範圍。完整次要外框產生後，才用原始參數拆成後續碎片；後續拆片不再溢出。\n\n勾選「反轉」可交換透明與不透明範圍。":
        "The secondary mask can be used alone. With a primary mask it processes only the primary interior; without one it processes the source image directly. The secondary interior is retained in the first item, while the secondary frame is split into later fragments. When Overflow is enabled, the secondary frame extends into the retained interior only during this separation. It is enabled by default and uses half the current block size and randomness (minimum 1). Disable it to strictly limit output to the secondary-frame area. After the complete secondary frame is built, the original settings split it into later fragments with no further overflow.\n\nInvert swaps transparent and opaque mask areas.",
    "固定處理順序：先由主體遮罩把主圖嚴格拆成外框與內部（不溢出）；再由次要遮罩處理這張內部圖；最後才拆外框碎片、產生干擾及合成第一張。\n\n外框與次要遮罩保留區會在處理完成後合進清單最上方碎片；初次拆解的名稱仍為碎片 1～N，共 N 張。\n\n遮罩必須是 PNG，且大小與主圖完全一致；alpha 大於 0 的像素視為遮罩範圍。只載入主體遮罩也可以執行。\n\n若兩張遮罩都不載入，則執行基本拆分；次要遮罩不能單獨使用。":
        "Fixed order: the primary mask strictly separates the source into frame and interior (no overflow). The secondary mask then processes that interior. Only afterward is the frame split, interference generated, and the first list item composed.\n\nThe primary frame and secondary retained area are combined into the top list item. Initial output remains Fragment 1–N, for a total of N items.\n\nMasks must be PNG files matching the source dimensions. Pixels with alpha above 0 are inside the mask. A primary mask may be used alone.\n\nWith no masks, a basic split is performed. A secondary mask cannot be used alone.",
    "固定處理順序：先由主體遮罩把主圖拆成外框與內部，再由次要遮罩處理這張內部圖；最後才拆外框碎片、產生干擾及合成第一張。勾選「溢出」時，只有主體外框會在這次主要內／外分離時向主要內部延伸；預設關閉。\n\n外框與次要遮罩保留區會在處理完成後合進清單最上方碎片；初次拆解的名稱仍為碎片 1～N，共 N 張。\n\n遮罩必須是 PNG，且大小與主圖完全一致；alpha 大於 0 的像素視為遮罩範圍。只載入主體遮罩也可以執行。\n\n若兩張遮罩都不載入，則執行基本拆分；次要遮罩不能單獨使用。":
        "Fixed order: the primary mask separates the source into frame and interior, then the secondary mask processes that interior. Only afterward are later fragments split, interference generated, and the first list item composed. When Overflow is enabled, only the primary frame extends into the primary interior during this separation; it is disabled by default.\n\nThe primary frame and secondary retained area are combined into the top list item. Initial output remains Fragment 1–N, for a total of N items.\n\nMasks must be PNG files matching the source dimensions. Pixels with alpha above 0 are inside the mask. A primary mask may be used alone.\n\nWith no masks, a basic split is performed. A secondary mask cannot be used alone.",
    "次要遮罩只處理主體遮罩拆出的內部圖。其次要內部不溢出，會保留到最後與主體外框合成第一張；次要外框只在這次內／外分離時允許溢進次要內部。這次溢出會使用方塊尺寸與隨機度各一半的參數（最低為 1）。完整次要外框產生後，才用原始參數拆成後續碎片；後續拆片不再溢出。\n\n次要遮罩不能單獨使用。勾選「反轉」可交換透明與不透明範圍。":
        "The secondary mask processes only the interior produced by the primary mask. Its retained interior never overflows and is later combined with the primary frame as the first list item. The secondary frame may overflow into the retained interior only during this inner/outer separation. That overflow uses half the current block size and randomness (minimum 1). After the complete secondary frame is built, the original settings split it into later fragments with no further overflow.\n\nA secondary mask cannot be used alone. Invert swaps transparent and opaque mask areas.",
    "次要遮罩只處理主體遮罩拆出的內部圖。其次要內部不溢出，會保留到最後與主體外框合成第一張。勾選「溢出」時，次要外框只在這次內／外分離時溢進次要內部；預設開啟，並使用方塊尺寸與隨機度各一半的參數（最低為 1）。關閉後會嚴格限制在次要外框範圍。完整次要外框產生後，才用原始參數拆成後續碎片；後續拆片不再溢出。\n\n次要遮罩不能單獨使用。勾選「反轉」可交換透明與不透明範圍。":
        "The secondary mask processes only the interior produced by the primary mask. Its retained interior never overflows and is later combined with the primary frame as the first list item. When Overflow is enabled, the secondary frame extends into the retained interior only during this separation. It is enabled by default and uses half the current block size and randomness (minimum 1). Disable it to strictly limit output to the secondary-frame area. After the complete secondary frame is built, the original settings split it into later fragments with no further overflow.\n\nA secondary mask cannot be used alone. Invert swaps transparent and opaque mask areas.",
    "此數值就是最終碎片總數。設定 6 時，清單與匯出結果為碎片 1、2、3、4、5、6。":
        "This is the final fragment count. A value of 6 produces Fragment 1 through Fragment 6 in the list and exports.",
    "定義分割的最小區塊（鏤空最小洞）的尺寸。數字越大，每個分割塊越大。單位：px\n\n優點：區塊大可提升運算速度、減少碎片數。\n\n缺點：太大會降低隱蔽度，過小可能造成卡頓。":
        "Minimum split-block (smallest cutout) size in pixels. Larger values produce larger blocks.\n\nBenefit: faster processing and fewer regions.\n\nTrade-off: large blocks reduce concealment; very small blocks may cause slowdowns.",
    "區塊尺寸的隨機倍率範圍，1 代表所有區塊尺寸固定，2 代表區塊尺寸會隨機在設定值的 1~2 倍間變化。\n\n優點：提高碎片形狀隨機性，難以預測與還原。\n\n缺點：過高會造成計算量大增與碎片難以辨認。":
        "Random multiplier for split-block size. 1 keeps every block fixed; 2 varies blocks between 1× and 2× the base size.\n\nBenefit: less predictable shapes.\n\nTrade-off: high values increase work and make fragments harder to inspect.",
    "拆解後於鏤空區補原圖像素作為重疊像素。\n數值為聯集不透明像素的比例，依各碎片可填補區域分別回補。\n\n優點：增加還原難度，讓每片有干擾。\n\n缺點：比例過高會導致效能大幅下降、檔案變大。":
        "Fills transparent cutouts with source pixels after splitting. The value is a percentage of the union of opaque pixels and is applied to each fragment's available fill area.\n\nBenefit: makes reconstruction harder.\n\nTrade-off: high values reduce performance and increase file size.",
    "調整回補的重疊像素聚集程度。1=最分散，10=最密集，預設5。\n\n優點：可調整碎片間重疊區域型態，提升反逆向性。\n\n缺點：極端值可能造成運算異常或不自然分佈。":
        "Controls clustering of filled overlap pixels. 1 is most dispersed, 10 most clustered; default is 5.\n\nBenefit: varies overlap patterns.\n\nTrade-off: extreme values may look unnatural or process poorly.",
    "可輸入範圍：1～10。此數值就是最終碎片總數。設定 6 時，清單與匯出結果為碎片 1、2、3、4、5、6。":
        "Input range: 1–10. This is the final fragment count. A value of 6 produces Fragment 1 through Fragment 6 in the list and exports.",
    "可輸入範圍：1～30 px。定義分割的最小區塊（鏤空最小洞）的尺寸。數字越大，每個分割塊越大。\n\n優點：區塊大可提升運算速度、減少碎片數。\n\n缺點：太大會降低隱蔽度，過小可能造成卡頓。":
        "Input range: 1–30 px. Sets the minimum split-block (smallest cutout) size. Larger values produce larger blocks.\n\nBenefit: faster processing and fewer regions.\n\nTrade-off: large blocks reduce concealment; very small blocks may cause slowdowns.",
    "可輸入範圍：1～100。區塊尺寸的隨機倍率範圍，1 代表所有區塊尺寸固定，2 代表區塊尺寸會隨機在設定值的 1～2 倍間變化。\n\n優點：提高碎片形狀隨機性，難以預測與還原。\n\n缺點：過高會造成計算量大增與碎片難以辨認。":
        "Input range: 1–100. Random multiplier for split-block size. 1 keeps every block fixed; 2 varies blocks between 1× and 2× the base size.\n\nBenefit: less predictable shapes.\n\nTrade-off: high values increase work and make fragments harder to inspect.",
    "可輸入範圍：0～100%。拆解後於鏤空區補原圖像素作為重疊像素。\n數值為聯集不透明像素的比例，依各碎片可填補區域分別回補。\n\n優點：增加還原難度，讓每片有干擾。\n\n缺點：比例過高會導致效能大幅下降、檔案變大。":
        "Input range: 0–100%. Fills transparent cutouts with source pixels after splitting. The value is a percentage of the union of opaque pixels and is applied to each fragment's available fill area.\n\nBenefit: makes reconstruction harder.\n\nTrade-off: high values reduce performance and increase file size.",
    "可輸入範圍：1～10，預設 5。調整回補的重疊像素聚集程度；1 為最分散，10 為最密集。\n\n優點：可調整碎片間重疊區域型態，提升反逆向性。\n\n缺點：極端值可能造成運算異常或不自然分佈。":
        "Input range: 1–10; default: 5. Controls clustering of filled overlap pixels. 1 is most dispersed and 10 is most clustered.\n\nBenefit: varies overlap patterns.\n\nTrade-off: extreme values may look unnatural or process poorly.",
    "可輸入範圍：1～30 px，預設 1 px。設定每一個干擾像素塊的基本邊長，越大則每塊越大。\n\n優點：大尺寸提升覆蓋速度。\n\n缺點：塊太大時，干擾效果會不自然且容易被辨識。":
        "Input range: 1–30 px; default: 1 px. Sets the base edge length of each interference block. Larger values create larger blocks.\n\nBenefit: faster coverage.\n\nTrade-off: very large blocks can look unnatural and become easier to identify.",
    "可輸入範圍：1～100，預設 6。決定干擾像素塊的尺寸隨機變動範圍，1 為固定，數字越大越亂。\n\n優點：隨機性高提升防還原性。\n\n缺點：數值過大會產生極端尺寸、不均勻塊。":
        "Input range: 1–100; default: 6. Controls interference-block size variation. 1 is fixed; higher values increase variation.\n\nBenefit: less predictable fragments.\n\nTrade-off: extreme values produce uneven block sizes.",
    "設定每一個干擾像素塊的基本邊長(px)，越大則每塊越大。\n\n優點：大尺寸提升覆蓋速度。\n\n缺點：塊太大時，干擾效果會不自然且容易被辨識。":
        "Sets the base edge length (px) of each interference block. Larger values create larger blocks.\n\nBenefit: faster coverage.\n\nTrade-off: very large blocks can look unnatural and become easier to identify.",
    "決定干擾像素塊的尺寸隨機變動範圍，1為固定，數字越大越亂。\n\n優點：隨機性高提升防還原性。\n\n缺點：數值過大會產生極端尺寸、不均勻塊。":
        "Controls interference-block size variation. 1 is fixed; higher values increase variation.\n\nBenefit: less predictable fragments.\n\nTrade-off: extreme values produce uneven block sizes.",
    "決定干擾像素填滿目標區域的比例，數字越高，干擾覆蓋越密集，100%不代表全填滿，實際上會受到區塊尺寸影響。\n\n優點：密度高可大幅阻礙還原。\n\n缺點：太高會讓檔案龐大且難以正常辨識。":
        "Controls the target coverage ratio. Higher values create denser interference; 100% may still leave gaps because block size affects placement.\n\nBenefit: harder reconstruction.\n\nTrade-off: high density increases file size and can hurt readability.",
    "設定可被選入干擾素材池的像素塊，必須覆蓋的最小不透明比例，避免選到太透明的雜訊。\n\n優點：濾除雜訊，保證干擾有效。\n\n缺點：設定過高會排除大部分素材，干擾池不足。":
        "Minimum opaque-area ratio required for a block to enter the sample pool. This rejects nearly transparent noise.\n\nTrade-off: a high value may leave too few samples.",
    "設定可被選入干擾素材池的像素塊，必須覆蓋的最大不透明比例。可用來排除太實心的大片塊。\n\n優點：排除過大塊避免影響外觀。\n\n缺點：過小則素材有限，干擾效果變差。":
        "Maximum opaque-area ratio allowed in the sample pool. Use it to reject overly solid blocks.\n\nTrade-off: a low value limits available samples.",
    "允許多個干擾像素塊彼此重疊。若關閉，干擾像素會盡量不交錯，但可能會減少填充面積。\n\n優點：允許重疊可提升覆蓋效率與密度。\n\n缺點：重疊過多時，部分區塊可能異常突出。":
        "Allows interference blocks to overlap. When disabled, blocks avoid one another where possible, which may reduce coverage.\n\nTrade-off: excessive overlap can make some areas stand out.",
    "勾選後，只把所選前方碎片完全不透明的區域加入生成範圍。\n取消勾選時，半透明像素也會加入所選碎片的聯集範圍。\n建議開啟，能避免在主圖透明邊緣產生髒點":
        "When enabled, only fully opaque pixels from selected earlier fragments enter the generation scope. Disable it to include semi-transparent pixels. Keeping it enabled avoids dirty transparent edges.",
    "干擾像素取自目前主圖；若劣化處理頁已掛載劣化取樣圖，則優先取自該劣化圖。清單第一張不加入干擾，也不會被拿來當生成範圍；第二張沒有前方可用範圍，保持原樣。從清單第三張開始，只會在第二張到目前碎片前一張之間隨機選擇一片或多片，將其 alpha 聯集作為干擾生成範圍。":
        "Interference is sampled from the current source image, or from the mounted degraded image when available. The first list item receives no interference and is never a generation scope. The second has no eligible earlier scope and remains unchanged. From the third item onward, one or more items between the second and the current item's predecessor are selected at random; their combined alpha defines the generation scope.",
    "此區用來製作干擾像素的劣化取樣圖。\n\n先匯入來源圖並調整下方劣化參數，再按「產生劣化預覽」；確認效果後按「掛載干擾像素」，全圖拆解與局部分割便會自動改用這張劣化圖取樣。\n\n「還原原圖」只把左側預覽切回匯入的原始來源，不會解除已掛載的干擾像素；載入新主圖、匯入新來源或重新產生劣化預覽時，掛載狀態會自動重設。":
        "Creates a degraded sampling image for interference.\n\nLoad a source, adjust the degradation settings, then choose Preview. After confirming the result, choose Mount Sample; full and area splitting will sample from the degraded image automatically.\n\nShow Original only changes the left preview and does not unmount the sample. Loading a new main image or source, or regenerating the preview, resets the mounted state.",
    "此區用來製作干擾像素的劣化取樣圖。\n\n先匯入來源圖並調整下方劣化參數，再按「產生劣化預覽」；確認效果後按「掛載干擾像素」，全圖拆解與局部分割便會自動改用這張劣化圖取樣。\n\n若要直接產生劣化碎片，可調整參數後按「製作劣化碎片」，不必先產生預覽。\n\n「還原原圖」只把左側預覽切回匯入的原始來源，不會解除已掛載的干擾像素；載入新主圖、匯入新來源或重新產生劣化預覽時，掛載狀態會自動重設。":
        "Creates a degraded sampling image for interference.\n\nLoad a source, adjust the degradation settings, then choose Preview. After confirming the result, choose Mount; full and area splitting will sample from it automatically.\n\nTo directly create degraded fragments, adjust the settings and choose Make Degraded Fragments; no preview is required.\n\nOrig. only changes the left preview and does not unmount the sample. Loading a new main image or source, or regenerating the preview, resets the mounted state.",
    "劣化方塊的基本尺寸（px），整張圖會以變動大小的方塊切割後個別劣化。":
        "Base degradation block size in pixels. The image is divided into varying blocks and each block is degraded independently.",
    "使用目前劣化預覽作為拆解來源，依現有遮罩與拆解參數產生碎片；完成後只在碎片拼接邊界隨機加入少量 1 px 縫隙與 1 px 局部錯位。結果會直接放入碎片管理，原始主圖不會被取代。":
        "Uses the current degraded preview as the split source with the active masks and split settings. It then adds sparse 1 px gaps and local 1 px offsets only along fragment seams. Results are placed directly in Fragments without replacing the original source image.",
    "直接依目前劣化參數處理已匯入的來源圖，不必先產生劣化預覽；接著依現有遮罩與拆解參數產生碎片。完成後會沿碎片外輪廓隨機加入較多的 1 px 缺口線段與 1 px 錯位線段。結果會直接放入碎片管理，原始主圖不會被取代。":
        "Directly applies the current degradation settings to the imported source; no preview is required. It then uses the active masks and split settings. More 1 px gap segments and 1 px offset segments are added along fragment outlines. Results go directly to Fragments without replacing the original source.",
    "直接依目前劣化參數處理已匯入的來源圖，不必先產生劣化預覽；接著依現有遮罩與拆解參數產生碎片。完成後會沿碎片外輪廓加入 1 px 缺口與錯位線段，合計影響約 20% 的拼接輪廓。結果會直接放入碎片管理，原始主圖不會被取代。":
        "Directly applies the current degradation settings to the imported source; no preview is required. It then uses the active masks and split settings. Combined 1 px gap and offset segments affect about 20% of fragment seams. Results go directly to Fragments without replacing the original source.",
    "直接依目前劣化參數處理已匯入的來源圖，不必先產生劣化預覽；接著依現有遮罩與拆解參數產生碎片。完成後會沿碎片外輪廓加入 1 px 缺口與錯位線段，合計影響約 20% 的拼接輪廓；1 px 縫隙會用原像素的 60% Alpha 補齊；每張碎片本身也會向隨機方向整體錯位 1 px，空位會用原邊緣像素延展補齊。結果會直接放入碎片管理，原始主圖不會被取代。":
        "Directly applies the current degradation settings to the imported source; no preview is required. It then uses the active masks and split settings. Combined 1 px gap and offset segments affect about 20% of fragment seams; the original pixels are restored there at 60% alpha. Every fragment is also shifted as a whole by 1 px in a random direction, and vacated areas are filled by extending the original edge pixels. Results go directly to Fragments without replacing the original source.",
    "控制整張圖中要放多少塊進行劣化（影響劣化區塊數量）。":
        "Controls how many blocks across the image are degraded.",
    "每個方塊中加入的隨機雜訊強度。": "Amount of random noise added to each block.",
    "每個方塊會有明暗偏移。": "Applies a random brightness shift to each block.",
    "每個方塊加入隨機 RGB 色偏。": "Applies a random RGB color shift to each block.",
    "劣化方塊尺寸的隨機倍率範圍，1 代表所有區塊尺寸固定，2 代表區塊尺寸會隨機在設定值的 1~2 倍間變化。":
        "Random multiplier for degradation block size. 1 keeps blocks fixed; 2 varies them from 1× to 2× the base size.",
})


EXACT_EN.update({
    # Validation, confirmations and progress text.
    "高風險參數警告": "High-Risk Settings",
    "方塊尺寸極小且重疊像素比例高，會嚴重卡頓甚至當機！":
        "A very small block size with high overlap may freeze or crash the application.",
    "方塊尺寸小於等於2，會產生極大量碎片，容易造成當機。":
        "A block size of 2 or less creates a very large number of regions and may cause a crash.",
    "碎片數量超過20，極易造成記憶體暴增與當機。":
        "More than 20 fragments may sharply increase memory use and cause a crash.",
    "尺寸隨機度過高且方塊太小，碎片組合將暴增，容易當機。":
        "High size randomness with small blocks greatly increases processing work and may cause a crash.",
    "重疊像素比例超過20%，處理大圖或高分割時可能造成介面無回應或記憶體不足。":
        "Overlap above 20% may make large or heavily split images unresponsive or exhaust memory.",
    "重疊像素聚合度高且比例大於5%，會讓補丁集中、容易卡死。":
        "High clustering with overlap above 5% concentrates patches and may make the application unresponsive.",
    "方塊尺寸與隨機度相乘過大，將產生異常碎片，容易當機。":
        "An excessive block-size/randomness product may create abnormal fragments and cause a crash.",
    "\n\n確定要繼續執行嗎？": "\n\nContinue?",
    "當前碎片管理中還有碎片。\n\n執行拆解會把這些碎片全數移到垃圾桶，確定要繼續嗎？":
        "Fragments are already present.\n\nStarting a new split moves all of them to Trash. Continue?",
    "次要遮罩必須搭配主體遮罩；請載入主體遮罩或移除次要遮罩":
        "A secondary mask requires a primary mask. Load a primary mask or remove the secondary mask.",
    "次要遮罩不能單獨用於局部分割；請載入主體遮罩或移除次要遮罩。":
        "A secondary mask cannot be used alone for area splitting. Load a primary mask or remove the secondary mask.",
    "局部分割將覆蓋既有干擾像素": "Area Split Replaces Existing Interference",
    "局部分割會先清除框選範圍內既有碎片內容，再依目前分片數量重新分配。\n\n重新分割完成後，程式會只在框選範圍內自動套用目前的干擾像素參數；框選範圍外的既有內容與干擾不會改變。":
        "Area splitting first clears existing fragment content inside the selection, then redistributes it using the current fragment count.\n\nAfterward, the current interference settings are applied only inside the selection. Existing content and interference outside it remain unchanged.",
    "請在左側預覽用滑鼠右鍵拖曳框出一個區域後再按此鍵":
        "Right-drag in the left preview to select an area before using this command.",
    "分片產生失敗，請檢查遮罩與參數": "Fragment generation failed. Check the masks and settings.",
    "分片產生失敗，請檢查遮罩與拆解參數": "Fragment generation failed. Check the masks and split settings.",
    "選定區域內沒有可分割的像素": "The selected area contains no splittable pixels.",
    "處理中...": "Processing...",
    "執行拆解進度": "Split Progress",
    "局部分割進度": "Area Split Progress",
    "局部分割：正在建立外框與碎片...": "Area split: building frame and fragments...",
    "局部分割：正在產生干擾像素...": "Area split: generating interference...",
    "執行拆解：正在填充重疊像素...": "Split: filling overlap pixels...",
    "執行拆解：正在產生干擾像素...": "Split: generating interference...",
    "一鍵拆解：正在填充重疊像素...": "Auto split: filling overlap pixels...",
    "一鍵拆解：正在以主圖產生干擾...": "Auto split: generating interference from the source image...",
    "一鍵拆解：正在以掛載劣化圖產生干擾...": "Auto split: generating interference from the mounted degraded image...",
    "重疊像素處理失敗，改用原始分割結果": "Overlap processing failed; using the original split result.",
    "局部分割完成，但沒有可處理的碎片": "Area split completed, but there are no fragments to process.",
    # File, fragment and preview operations.
    "主體遮罩已重新載入": "Primary mask reloaded.",
    "次要遮罩已重新載入": "Secondary mask reloaded.",
    "已移除主體遮罩": "Primary mask removed.",
    "已移除次要遮罩": "Secondary mask removed.",
    "已取消執行": "Operation cancelled.",
    "已取消拆解": "Split cancelled.",
    "已取消劣化預覽": "Degradation preview cancelled.",
    "請先載入主圖，並有碎片可預覽": "Load a source image and create fragments before using this preview.",
    "沒有任何碎片可匯出！": "There are no fragments to export.",
    "請先選取要匯出的碎片！": "Select fragments to export first.",
    "合併至少需要選取兩個碎片": "Select at least two fragments to merge.",
    "請先選取要複製的碎片": "Select fragments to duplicate.",
    "請先選取要刪除的碎片": "Select fragments to delete.",
    "請先選取要重新命名的碎片": "Select fragments to rename.",
    "此名稱已存在，請選擇其他名稱。": "That name already exists. Choose another name.",
    "請輸入前綴（例如：碎片）": "Enter a prefix (for example, Fragment)",
    "所有碎片皆已隱藏": "All fragments are hidden.",
    # Degradation and the retained legacy messages.
    "匯入劣化來源圖": "Load Degradation Source",
    "重新產生劣化預覽": "Regenerate Preview",
    "目前已有劣化預覽。重新產生會解除現有掛載並取代預覽，確定要繼續嗎？":
        "A degradation preview already exists. Regenerating it unmounts and replaces the current preview. Continue?",
    "請先產生劣化預覽": "Generate a degradation preview first.",
    "已掛載劣化圖；後續拆解將用它產生干擾像素": "Degraded sample mounted for subsequent splitting.",
    "至少需要兩片碎片才能產生干擾": "At least two fragments are required to generate interference.",
    "找不到第一片遮罩碎片": "The first scope fragment could not be found.",
    "所有碎片必須使用相同畫布尺寸才能產生干擾。\n請先處理以下碎片：":
        "All fragments must use the same canvas size. Check these fragments first:",
    "請先載入要作為干擾像素來源的主圖（可使用劣化處理後的圖片）。":
        "Load the source image used for interference sampling. A degraded image may also be used.",
    "沒有產生任何干擾像素，請調整參數後再試。": "No interference was generated. Adjust the settings and try again.",
    "請先產生干擾像素": "Generate interference first.",
    "找不到碎片": "Fragments could not be found.",
    # PSD errors and worker messages.
    "PSD 檔案仍在背景儲存，請等待進度完成後再關閉程式。":
        "The PSD is still being saved in the background. Wait for completion before closing the application.",
    "尚未安裝 PSD 匯出套件，請先執行：pip install psd-tools>=1.10":
        "PSD export support is not installed. Run: pip install psd-tools>=1.10",
    "PSD 匯出已取消": "PSD export cancelled.",
    "劣化中...": "Degrading...",
    "主圖必須是 RGBA 圖片": "The source image must be RGBA.",
    "主體遮罩尺寸必須與主圖一致": "The primary mask must match the source dimensions.",
    "次要遮罩尺寸必須與主圖一致": "The secondary mask must match the source dimensions.",
    "主體遮罩必須是 RGBA 圖片": "The primary mask must be RGBA.",
    "次要遮罩必須是 RGBA 圖片": "The secondary mask must be RGBA.",
    "主體遮罩沒有覆蓋主圖的任何不透明像素": "The primary mask covers no opaque source pixels.",
    "次要遮罩在主體遮罩內沒有任何有效範圍": "The secondary mask has no valid area inside the primary mask.",
    "主體遮罩內、次要遮罩外沒有可拆分的像素": "There are no splittable pixels inside the primary mask and outside the secondary mask.",
    "遮罩必須是單通道或 RGBA 圖片": "A mask must be single-channel or RGBA.",
})


# Dynamic status strings are translated by longest phrase first. Exact entries above
# always win, so these replacements only handle values such as counts and filenames.
PHRASE_EN = (
    ("MSW造型防盜拆解工具 專業版 ", ""),
    ("主體遮罩載入失敗", "Failed to load primary mask"),
    ("次要遮罩載入失敗", "Failed to load secondary mask"),
    ("主圖載入失敗", "Failed to load source image"),
    ("主體遮罩載入成功", "Primary mask loaded"),
    ("次要遮罩載入成功", "Secondary mask loaded"),
    ("主圖載入成功", "Source image loaded"),
    ("局部分割進度", "Area Split Progress"),
    ("局部分割完成", "Area split complete"),
    ("局部分割失敗", "Area split failed"),
    ("正在產生劣化預覽", "Generating degradation preview"),
    ("製作劣化碎片進度", "Degraded Fragment Progress"),
    ("劣化碎片完成", "Degraded fragments complete"),
    ("劣化碎片", "Degraded fragments"),
    ("分析碎片拼接邊界", "Analyzing fragment seams"),
    ("加入細微錯位效果", "Adding subtle offsets"),
    ("完成劣化碎片接縫", "Finishing degraded fragment seams"),
    ("正在加入細微接縫與錯位", "Adding subtle seams and offsets"),
    ("正在加入輪廓缺口與錯位線段", "Adding outline gap and offset segments"),
    ("加入輪廓錯位線段", "Adding outline offset segments"),
    ("完成輪廓缺口線段", "Finishing outline gap segments"),
    ("輪廓線段處理失敗", "Outline segment processing failed"),
    ("已加入輪廓缺口與錯位線段", "outline gap and offset segments added"),
    ("細微接縫處理失敗", "Subtle seam processing failed"),
    ("已加入細微接縫與錯位", "subtle seams and offsets added"),
    ("劣化來源圖", "degraded source"),
    ("已套用方案", "Plan applied"),
    ("已儲存方案", "Plan saved"),
    ("已刪除方案", "Plan deleted"),
    ("部分對應不存在", "some mappings are unavailable"),
    ("方案列數", "plan rows"),
    ("目前碎片", "current fragments"),
    ("劣化預覽（尚未掛載）", "Degradation preview (not mounted)"),
    ("已掛載劣化圖；後續拆解將用它產生干擾像素", "Degraded sample mounted for subsequent splitting"),
    ("已掛載干擾像素", "Sample mounted"),
    ("掛載干擾像素", "Mount sample"),
    ("已載入劣化來源圖", "Degradation source loaded"),
    ("請先匯入劣化來源圖", "Load a degradation source first"),
    ("顯示原始劣化來源圖", "Showing original degradation source"),
    ("干擾像素產生完成", "Interference generation complete"),
    ("開始產生干擾像素", "Generating interference"),
    ("未產生任何干擾像素", "No interference was generated"),
    ("建立共用主圖素材池", "Building shared source pool"),
    ("產生干擾像素中，已用", "Generating interference; elapsed"),
    (" 秒", " s"),
    ("重疊像素填充完成", "Overlap fill complete"),
    ("開始進行重疊像素填充", "Starting overlap fill"),
    ("正在填充重疊像素", "Filling overlap pixels"),
    ("重疊像素處理失敗", "Overlap processing failed"),
    ("分片產生失敗", "Fragment generation failed"),
    ("正在建立外框與碎片", "Building frame and fragments"),
    ("正在分配像素到碎片", "Assigning pixels to fragments"),
    ("正在產生分割區塊", "Generating split blocks"),
    ("產生碎片圖像", "Creating fragment images"),
    ("處理孤立像素", "Processing isolated pixels"),
    ("計算分割區塊", "Calculating split blocks"),
    ("建立次要外框的受控溢出範圍", "Building controlled secondary-frame overflow"),
    ("建立次要外框溢出範圍", "Building secondary-frame overflow"),
    ("開始分割", "Starting split"),
    ("拆解完成，花費", "Split complete in "),
    ("拆解中", "Splitting"),
    ("一鍵拆解", "Auto split"),
    ("已還原初始分割狀態", "Initial split restored"),
    ("尚未有分割過的結果可還原", "There is no split result to restore"),
    ("已退出進階管理", "Exited advanced mode"),
    ("進階管理預覽", "Advanced preview"),
    ("所有碎片皆已隱藏", "All fragments are hidden"),
    ("已匯出選擇碎片", "Selected fragments exported"),
    ("已匯出全部碎片到", "All fragments exported to"),
    ("壓縮匯出失敗", "ZIP export failed"),
    ("匯出失敗", "Export failed"),
    ("匯入失敗", "Import failed"),
    ("已匯入碎片", "Fragments imported"),
    ("已重新命名為", "Renamed to"),
    ("已批次命名為", "Batch renamed to"),
    ("已複製", "Duplicated"),
    ("已刪除", "Deleted"),
    ("已合併並移除原有碎片", "Merged and removed original fragments"),
    ("已復原碎片", "Fragment restored"),
    ("已儲存", "Saved"),
    ("儲存失敗", "Save failed"),
    ("垃圾桶預覽：已選擇", "Trash preview: selected"),
    ("垃圾碎片", "Trash fragments: "),
    ("點擊眼睛隱藏此碎片", "Click the eye to hide this fragment"),
    ("點擊眼睛顯示此碎片", "Click the eye to show this fragment"),
    ("主體遮罩預覽", "Primary-mask preview"),
    ("次要遮罩預覽", "Secondary-mask preview"),
    ("主圖預覽", "Source-image preview"),
    ("取樣不透明度下限", "Sample Alpha Min"),
    ("取樣不透明度上限", "Sample Alpha Max"),
    ("干擾密度", "Density"),
    ("劣化密度", "Density"),
    ("噪點強度", "Noise"),
    ("隨機明暗", "Brightness"),
    ("色偏強度", "Color Shift"),
    ("碎片預覽", "fragment preview"),
    ("干擾像素預覽中 尚未合成", "interference preview (not composited)"),
    ("劣化圖已掛載為干擾像素主圖", "Degraded image mounted as interference source"),
    ("翻找垃圾桶中", "Browsing trash"),
    ("正在背景匯出 PSD", "Exporting PSD in background"),
    ("PSD 匯出完成", "PSD export complete"),
    ("PSD 匯出失敗", "PSD export failed"),
    ("正在讀取 PSD 範本", "Reading PSD template"),
    ("已讀取 PSD 範本", "PSD template loaded"),
    ("正在寫入圖層", "Writing layer"),
    ("已寫入圖層", "Layer written"),
    ("正在儲存", "Saving"),
    ("已儲存", "Saved"),
    ("尚未選擇 PSD 檔案", "has no PSD file selected"),
    ("尚未選擇 PSD 圖層", "has no PSD layer selected"),
    ("找不到指定圖層，請重新選擇", "target layer was not found; select it again"),
    ("沒有可匯出的 PSD 設定", "No PSD export assignments"),
    ("匯出位置不能覆蓋專案內的 PSD 範本", "The output cannot overwrite a project PSD template"),
    ("檔名前綴不能以句點結尾", "The filename prefix cannot end with a period"),
    ("檔名前綴不能包含", "The filename prefix cannot contain"),
    ("未命名圖層", "Unnamed Layer"),
    ("碎片圖層", "Fragment Layer"),
    ("讀取失敗", "Read failed"),
    ("背景處理尚未完全停止，請稍候再關閉程式。", "Background work is still stopping. Please wait before closing the application."),
    ("請先載入主圖", "Load a source image first"),
    ("請先選取要重新命名的碎片", "Select fragments to rename"),
    ("請先選取要複製的碎片", "Select fragments to duplicate"),
    ("請先選取要刪除的碎片", "Select fragments to delete"),
    ("請先選取要匯出的碎片", "Select fragments to export"),
    ("找不到名為", "Cannot find a fragment named"),
    ("此名稱已存在，請選擇其他名稱。", "That name already exists. Choose another name."),
    ("至少需要兩片碎片", "At least two fragments are required"),
    ("選定區域內沒有可分割的像素", "The selected area contains no splittable pixels"),
    ("無法預覽", "cannot be previewed"),
    ("無可用範圍", "No available scope"),
    ("解析度", "Resolution"),
)


def set_language(language):
    global _language
    _language = LANG_EN if language == LANG_EN else LANG_ZH_TW


def current_language():
    return _language


def has_han(text):
    return bool(re.search(r"[\u3400-\u9fff]", str(text)))


def tr(text):
    if _language != LANG_EN or not isinstance(text, str) or not text:
        return text
    exact = EXACT_EN.get(text)
    if exact is not None:
        return exact
    match = re.fullmatch(r"垃圾碎片\s*(\d+)個", text)
    if match:
        return f"Trash fragments: {match.group(1)}"
    dynamic_percent_labels = {
        "干擾密度": "Density",
        "取樣不透明度下限": "Sample Alpha Min",
        "取樣不透明度上限": "Sample Alpha Max",
        "劣化密度": "Density",
        "噪點強度": "Noise",
        "隨機明暗": "Brightness",
        "色偏強度": "Color Shift",
    }
    match = re.fullmatch(r"(.+)[：:]\s*(\d+)%", text)
    if match and match.group(1) in dynamic_percent_labels:
        return f"{dynamic_percent_labels[match.group(1)]}: {match.group(2)}%"
    match = re.fullmatch(r"碎片\s*(\d+)", text)
    if match:
        return f"Fragment {match.group(1)}"
    match = re.fullmatch(r"(\d+)小時(\d+)分(\d+)秒", text)
    if match:
        return f"{match.group(1)}h {match.group(2)}m {match.group(3)}s"
    match = re.fullmatch(r"(\d+)分(\d+)秒", text)
    if match:
        return f"{match.group(1)}m {match.group(2)}s"
    match = re.fullmatch(r"(\d+)秒", text)
    if match:
        return f"{match.group(1)}s"
    match = re.fullmatch(r"垃圾桶預覽：已選擇\s*(\d+)\s*個碎片", text)
    if match:
        return f"Trash preview: {match.group(1)} selected"
    match = re.fullmatch(r"遮罩拆解：正在產生碎片 2～(\d+)\.\.\.", text)
    if match:
        return f"Mask split: creating Fragments 2–{match.group(1)}..."
    match = re.fullmatch(r"基本拆分：正在產生\s*(\d+)\s*張碎片\.\.\.", text)
    if match:
        return f"Basic split: creating {match.group(1)} fragments..."
    if text == "遮罩拆解：正在建立唯一碎片...":
        return "Mask split: creating the only fragment..."
    match = re.fullmatch(
        r"局部分割完成，正在為最上方\s*(\d+)\s*張套用局部干擾\.\.\.", text
    )
    if match:
        return f"Area split complete; applying interference to the top {match.group(1)} items..."
    match = re.fullmatch(
        r"局部分割完成：框選內容位於最上方\s*(\d+)\s*張，局部干擾已套用", text
    )
    if match:
        return f"Area split complete: the selection occupies the top {match.group(1)} items and interference was applied."
    match = re.fullmatch(
        r"局部分割完成，但局部干擾僅成功\s*(\d+)/(\d+)\s*張", text
    )
    if match:
        return f"Area split complete, but interference succeeded on only {match.group(1)}/{match.group(2)} targets."
    match = re.fullmatch(
        r"拆解完成：碎片 1～(\d+)，共\s*(\d+)\s*張；干擾來源為(.+)（(\d+)秒）",
        text,
    )
    if match:
        source = tr(match.group(3))
        return f"Split complete: Fragments 1–{match.group(1)}, {match.group(2)} total; source: {source} ({match.group(4)}s)"
    match = re.fullmatch(
        r"已產生\s*(\d+)\s*張碎片，但只有\s*(\d+)/(\d+)\s*張成功套用干擾",
        text,
    )
    if match:
        return f"Created {match.group(1)} fragments, but interference succeeded on only {match.group(2)}/{match.group(3)} targets."
    match = re.fullmatch(r"已複製\s*(\d+)\s*個碎片", text)
    if match:
        return f"Duplicated {match.group(1)} fragments."
    match = re.fullmatch(r"已刪除\s*(\d+)\s*個碎片", text)
    if match:
        return f"Deleted {match.group(1)} fragments."
    match = re.fullmatch(r"已合併並移除原有碎片，共(\d+)個->1", text)
    if match:
        return f"Merged {match.group(1)} fragments into one and removed the originals."
    match = re.fullmatch(r"(.+) 的干擾範圍：(.+)", text)
    if match:
        return f"{match.group(1)} scope: {match.group(2)}"
    match = re.fullmatch(r"產生 (.+) 干擾像素失敗: (.+)", text)
    if match:
        return f"Failed to generate interference for {match.group(1)}: {match.group(2)}"
    match = re.fullmatch(
        r"碎片「(.+)」尺寸為 (\d+)×(\d+)，但 PSD 尺寸為 (\d+)×(\d+)", text
    )
    if match:
        return f'Fragment "{match.group(1)}" is {match.group(2)}×{match.group(3)}, but the PSD is {match.group(4)}×{match.group(5)}.'
    match = re.fullmatch(r"遮罩尺寸 (.+) 與圖片尺寸 (.+) 不一致", text)
    if match:
        return f"Mask dimensions {match.group(1)} do not match image dimensions {match.group(2)}."
    match = re.fullmatch(r"已儲存方案「(.+)」，共 (\d+) 個碎片對應。", text)
    if match:
        return f'Saved plan "{match.group(1)}" with {match.group(2)} fragment mappings.'
    match = re.fullmatch(r"已刪除方案「(.+)」。", text)
    if match:
        return f'Deleted plan "{match.group(1)}".'
    match = re.fullmatch(r"已套用方案「(.+)」，共 (\d+) 個碎片對應。", text)
    if match:
        return f'Applied plan "{match.group(1)}" to {match.group(2)} fragments.'
    match = re.fullmatch(r"已套用方案「(.+)」的 (\d+) 列；部分對應不存在：(.+)", text)
    if match:
        return f'Applied {match.group(2)} rows from plan "{match.group(1)}"; unavailable mappings: {match.group(3)}'
    match = re.fullmatch(r"方案「(.+)」已存在，是否覆蓋？", text)
    if match:
        return f'Plan "{match.group(1)}" already exists. Overwrite it?'
    match = re.fullmatch(r"確定要刪除方案「(.+)」嗎？", text)
    if match:
        return f'Delete plan "{match.group(1)}"?'
    match = re.fullmatch(r"「(.+)」尚未完整選擇 PSD 檔案與圖層", text)
    if match:
        return f'"{match.group(1)}" does not have both a PSD file and layer selected.'
    translated = "\n".join(EXACT_EN.get(line, line) for line in text.split("\n"))
    for source, target in PHRASE_EN:
        translated = translated.replace(source, target)
    return translated


class UiLanguageFilter(QtCore.QObject):
    """Retranslate live/dynamic widgets and provide button hover/press feedback."""

    _TEXT_EVENT_TYPES = {
        QtCore.QEvent.Show,
        QtCore.QEvent.Polish,
        QtCore.QEvent.Paint,
        QtCore.QEvent.ToolTip,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._syncing = False

    def eventFilter(self, obj, event):
        event_type = event.type()
        if isinstance(obj, QtWidgets.QPushButton):
            if event_type == QtCore.QEvent.Enter and obj.isEnabled():
                self._set_button_state(obj, "hover")
            elif event_type == QtCore.QEvent.MouseButtonPress and obj.isEnabled():
                self._set_button_state(obj, "pressed")
            elif event_type == QtCore.QEvent.MouseButtonRelease and obj.isEnabled():
                self._set_button_state(obj, "hover" if obj.underMouse() else "normal")
            elif event_type in (QtCore.QEvent.Leave, QtCore.QEvent.EnabledChange):
                self._set_button_state(obj, "normal")

        if event_type in self._TEXT_EVENT_TYPES and not self._syncing:
            self.retranslate_object(obj, recursive=event_type in (QtCore.QEvent.Show, QtCore.QEvent.Polish))
        return False

    @staticmethod
    def _set_button_state(button, state):
        current_effect = button.graphicsEffect()
        own_effect = getattr(button, "_interaction_effect", None)
        if own_effect is None and current_effect is not None:
            button._interaction_original_effect = current_effect

        if state == "normal" or not button.isEnabled():
            if current_effect is own_effect:
                button.setGraphicsEffect(
                    getattr(button, "_interaction_original_effect", None)
                )
            button._interaction_effect = None
            return

        effect = QtWidgets.QGraphicsColorizeEffect(button)
        if state == "pressed":
            effect.setColor(QtGui.QColor("#12384d"))
            effect.setStrength(0.62)
        else:
            effect.setColor(QtGui.QColor("#72d1ff"))
            effect.setStrength(0.32)
        button._interaction_effect = effect
        button.setGraphicsEffect(effect)

    def retranslate_object(self, obj, recursive=True):
        if self._syncing or obj is None:
            return
        self._syncing = True
        try:
            self._sync_one(obj)
            if recursive and isinstance(obj, QtCore.QObject):
                for child in obj.findChildren(QtCore.QObject):
                    self._sync_one(child)
        finally:
            self._syncing = False

    def _sync_value(self, obj, key, current, setter):
        if not isinstance(current, str) or obj.property("i18n_skip"):
            return
        records = getattr(obj, "_i18n_records", None)
        if records is None:
            records = {}
            obj._i18n_records = records
        record = records.get(key)
        if record is None:
            record = {"source": current, "last": current}
            records[key] = record
        elif current != record["last"]:
            # Application code supplied a new Traditional-Chinese dynamic value.
            record["source"] = current
        target = tr(record["source"])
        if current != target:
            setter(target)
        record["last"] = target

    def _sync_one(self, obj):
        if isinstance(obj, QtWidgets.QAbstractButton):
            self._sync_value(obj, "text", obj.text(), obj.setText)
        elif isinstance(obj, QtWidgets.QLabel):
            self._sync_value(obj, "text", obj.text(), obj.setText)
        elif isinstance(obj, QtWidgets.QGroupBox):
            self._sync_value(obj, "title", obj.title(), obj.setTitle)
        elif isinstance(obj, QtGui.QAction):
            self._sync_value(obj, "text", obj.text(), obj.setText)

        if isinstance(obj, QtWidgets.QLineEdit):
            self._sync_value(
                obj, "placeholder", obj.placeholderText(), obj.setPlaceholderText
            )
        if isinstance(obj, QtWidgets.QWidget):
            self._sync_value(obj, "window_title", obj.windowTitle(), obj.setWindowTitle)
            self._sync_value(obj, "tooltip", obj.toolTip(), obj.setToolTip)
        if isinstance(obj, QtGui.QAction):
            self._sync_value(obj, "tooltip", obj.toolTip(), obj.setToolTip)
            self._sync_value(obj, "status_tip", obj.statusTip(), obj.setStatusTip)
        if isinstance(obj, QtWidgets.QTabWidget):
            for index in range(obj.count()):
                self._sync_value(
                    obj,
                    f"tab_{index}",
                    obj.tabText(index),
                    lambda value, i=index: obj.setTabText(i, value),
                )
        if isinstance(obj, QtWidgets.QComboBox):
            for index in range(obj.count()):
                self._sync_value(
                    obj,
                    f"combo_{index}",
                    obj.itemText(index),
                    lambda value, i=index: obj.setItemText(i, value),
                )
        if isinstance(obj, QtWidgets.QTableWidget):
            for index in range(obj.columnCount()):
                item = obj.horizontalHeaderItem(index)
                if item is not None:
                    self._sync_value(
                        obj,
                        f"header_{index}",
                        item.text(),
                        lambda value, current=item: current.setText(value),
                    )
