# オンデバイス生成パイプライン計画(Pixel 10 Pro、2026-07-19)

> **状態(同日更新)**: ビュー合成の主経路は本書の検討を経て
> **CharacterGen**に確定し、統合計画は
> [`REAL_ART_LIVE2D_PLAN.md`](./REAL_ART_LIVE2D_PLAN.md)(v2)に移した。
> 本書はオンデバイス実行環境(Local Dream等)の調査記録および
> 仕上げチェーン(アップスケール+低ノイズimg2img)の根拠資料として残す。

> 「スマホ動く画像生成AIでもいい」を受けた具体化。前提は変えない:
> 原画ピクセル維持/正面図中心/VTuber・可愛さ・3Dゲームの3要件。
> ユーザー端末: **Google Pixel 10 Pro(Tensor G5、Snapdragon系NPU非搭載)**。
> 実装はまだ行わない(計画のみ)。
>
> **追記(同日)**: ユーザーがZero123のオンラインデモで本キャラ正面画像から
> 新規ビュー生成を試したところ、**画風(アニメ塗り)を保ったまま生成できる
> ことを実証済み**。当初の懸念(Zero123はObjaverse等の3D物体全般で学習
> されておりアニメ塗りでは画風が崩れるのでは)は外れた。よって中間ビュー
> 生成の主力手段は §3〜4 のSD1.5低ノイズimg2img経由ではなく、
> **Zero123系の直接的な新規ビュー合成を第一候補**に格上げする(§8参照)。

これまでの計画では「ローカル画像生成AI = サンドボックス内は不可能」と
結論していたが、**ユーザーのスマホ自体をローカル生成環境として使える**
ことを見落としていた。Pixel 10 ProはCPU/GPU推論でオンデバイスSDが動く。

## 2. 候補アプリの調査結果

| アプリ | 機能 | Pixel(Tensor)での速度 |
|---|---|---|
| [Local Dream](https://github.com/xororz/local-dream)([Google Play](https://play.google.com/store/apps/details?id=io.github.xororz.localdream)、v2.3.3・2026年3月) | txt2img/img2img/インペイント/LoRA読込/embeddings/upscaler | NPU高速モード(5〜10秒/枚)は**Snapdragon専用(QNN SDK)のため非対応**。CPU/GPUフォールバックで**数分/枚**が現実的見積もり |
| [Local-Diffusion](https://github.com/rmatif/Local-Diffusion) | SD1.5/2.x/3.x・SDXL・FLUX.1、ControlNet系統含む | 同様にCPU/GPU依存、低速だが動作 |

**結論**: 爆速NPUは使えないが、**数分/枚なら実用可能**。この計画で
必要なのは数百枚の量産ではなく数十枚のキーフレームのみ(中間ビュー
数枚+表情差分十数枚)なので、就寝前などにまとめて仕込む運用で十分足りる。

## 3. 同一性ガチャへの対策: 低ノイズimg2img + Claude製下絵

Pixel側にControlNetが確実には無い前提でも、**img2imgの低ノイズ運用
(denoise 0.3〜0.4程度)** で同等の効果が出せる:

```
1. Claudeが既存4面(front/back/side_left/side_right)から
   ビューモーフィング(対応点ベースの決定的ワープ+ブレンド)で
   中間ビューの「下絵」を合成する。
   構図・配色・プロポーションは原画ピクセル由来で固定済み。
   継ぎ目や隠れ部分の欠損だけが粗い状態。
2. Pixel 10 Pro上のLocal Dreamで、その下絵をimg2img(低ノイズ)に
   かける。アニメ系SD1.5モデルが継ぎ目と欠損だけを「清書」する。
   低ノイズなので構図・輪郭は下絵に強く拘束され、
   同一性が崩れるガチャがほぼ起きない。
3. リポジトリに戻し、Claudeが目視QC。必要なら目・口領域だけ
   原画/Gemini差分から再合成して局所的な同一性をさらに補強する。
```

同一性を担保する主役は常にステップ1の下絵(=原画ピクセル)であり、
スマホAIは質感・継ぎ目の後処理係という位置づけに徹させる。これが
「全体をAIに再生成させる」方式との決定的な違いで、既存の防衛線思想
(局所差分だけAI、他は原画固定)と一貫している。

## 4. アセット供給の最終序列(更新版)

```
錨ビュー4面(front/back/side_left/side_right)
  = 既存・検収済み・今後も動かさない

中間ビュー(45°刻みなど)
  = Claudeのビューモーフィング下絵 → Pixel Local Dreamでimg2img清書
    → Claude目視QC

局所差分(表情・口形状)
  = 同じ経路(下絵+低ノイズimg2img)、または軽微ならGeminiの
    局所差分編集でも可(どちらも「原画拘束・全体再生成しない」原則を守る)

将来オプション: LoRA学習
  = Local DreamはLoRA読込対応。量が増え同一性が不安定になったら、
    このキャラ専用LoRAをPixel上またはGPU環境で学習し、
    低ノイズimg2imgの拘束をさらに強化する
```

外部クラウド生成(Gemini)への依存は「あれば便利」まで後退し、
**オフラインでも完結できる経路が主線になる**。

## 5. 残る不確実性(正直な記述)

- Pixel(Tensor)でのLocal Dream実測速度・品質は未検証(調査は公開情報
  ベース)。実際に触ってみないと「数分/枚」の精度は確定しない。
- ビューモーフィングの下絵品質(対応点の精度)が低ノイズimg2imgの
  効き目を左右する。下絵が粗いと清書後も違和感が残る可能性がある。
- SD1.5系アニメモデルの選定(どのcheckpointを使うか)は未決定。
  ユーザー側でモデルファイルを入手・配置する手間が発生する。

## 6. 着手順(提案、ユーザー指示待ち)

1. Pixel 10 ProにLocal Dreamを導入し、適当なアニメ系SD1.5モデルで
   txt2img/img2imgの実速度・品質を軽く試す(実測値の確定)
2. Claude側でビューモーフィングの最小実証
   (front→side_leftの1区間、45°相当を1枚合成)
3. 1の下絵を2でPixelのimg2imgにかけ、清書結果を目視評価
4. 通れば中間ビュー全区間・表情差分へ展開

## 7. Zero123実証を受けた更新(2026-07-19追記)

### 7.1 何が変わったか

- ビューモーフィング(§3ステップ1、対応点ベースの決定的ワープ)は
  「2枚の実写が無いと使えない」「隠れた部分は欠損する」という制約が
  あったが、**Zero123は正面1枚から新規ビューを直接合成できる**ことが
  実証されたため、中間ビュー生成の主経路として優先する。
- ビューモーフィングは**Zero123が使えない/品質が出ない場合の代替**、
  または**Zero123出力の後処理(継ぎ目補正)用の下絵**として温存する
  (完全に破棄はしない)。

### 7.2 更新後のアセット供給序列

```
錨ビュー4面(front/back/side_left/side_right)
  = 既存・検収済み・今後も動かさない

中間ビュー(45°刻みなど)・将来キャラの新規ビュー
  = 正面(または既存ビュー)からZero123で直接合成
    → Claude目視QC(既存4面と整合するか、画風が崩れていないか)
    → 通らなければビューモーフィング下絵+img2img清書(§3)に切替

局所差分(表情・口形状)
  = 引き続きGeminiの局所差分編集、または低ノイズimg2img経由
    (Zero123は新規ビュー合成が専門で、表情差分向きではないため)
```

### 7.3 残る検証課題

- 使用したのは**オンラインデモ**であり、Pixel 10 Pro上でオフライン
  完結できるかは未検証(Zero123系はSD1.5よりモデルサイズ・計算量が
  大きく、ブラウザWebGPUや Local Dream 系アプリでの実行実績も未確認)。
  オフライン化できない場合、この経路は「Gemini同様、あれば便利な
  クラウド/デモ依存」の位置づけに留まる。
- 生成された新規ビューが**既存4面(特にback/side)と厳密に整合するか**
  (プロポーション・配色・柄の向きなど)はまだ目視QCしていない。
  中間ビューとして採用する前に、既存4面との突き合わせが必要。
- 表情・口形状のような細部差分にZero123が向くかは未検証(向いていない
  可能性が高く、その場合はGeminiとの役割分担がそのまま生きる)。

## 8. アプリ・モデル比較の深掘り(2026-07-19追記、Pixel/サンドボックス双方を再検討)

### 8.1 Pixel 10 Pro側: アプリ比較(Local Dream/Local-Diffusionに加え再検討)

| アプリ | img2img/denoise制御 | Pixel(Tensor)適性 | 評価 |
|---|---|---|---|
| Off Grid | txt2img中心、モデルブラウザ経由で制御が弱い | Snapdragon NPU専用高速化、Tensorは他機種同様CPUフォールバック | §3の低ノイズimg2img清書には不向き |
| **Local Dream** | img2img専用UI・denoise strength明示、カスタムSD1.5/LoRA対応 | 同上(CPUフォールバック) | **候補筆頭**。img2img運用ノウハウの実績が厚い |
| SDAI | img2img/Inpaint/LoRA等プロ向け項目が明示 | 同上 | 次点。外部プロバイダ(Automatic1111等)併用への拡張性で勝る |

結論: **Local Dreamを軸に、行き詰まればSDAIへ切替**。Off Gridは今回の用途(顔まわり低ノイズimg2im清書)には不採用。

### 8.2 モデル比較(Pixel/サンドボックス共通の土台)

| モデル | 系統 | img2img低ノイズでの同一性保持の実績 | 判定 |
|---|---|---|---|
| Anima (2B) | Cosmos-Predict2ベース、2026年5月正式リリース | 新しすぎて未検証(GPU6GB前提の情報が中心) | 保留(要検証) |
| Animagine XL 4.0 Opt / Illustrious XL系(NoobAI-XL・Nova Anime XL等) | SDXL | 品質は最上位だがCPU/Pixelどちらでもステップ単価が重い | 不採用(RAM/速度の実用域外) |
| **SD1.5系(Anything V5 / MeinaMix)** | 旧世代 | img2img・低denoiseでの同一性保持の実績が最も厚い | **採用** |

結論: 品質最上位のIllustrious XL系はSDXLゆえPixel/サンドボックスどちらのローカル実行でも実用域外のため不採用。Anima 2Bは新しすぎて③(同一性保持)の実績が無いため保留。**Anything V5(またはMeinaMix)を軸に採用**。

### 8.3 サンドボックス側の再検討: 「不可能」から条件付き実行可能へ

本書§1(および旧結論)は「ローカル画像生成AI = サンドボックス内は不可能」としていたが、これは通常ステップ数(20〜50)のCPU推論を前提にした結論だった。実機確認(2026-07-19、本文書と同日): Intel Xeon 4コア・RAM15GB・GPU無し・ディスク空き30GB・非活動で回収。

- **OpenVINO+蒸留(LCM/Turbo)モデル**の組み合わせなら見直しの余地がある。OpenVINO CPUプラグインの最低要件はAVX2で、データセンター向けXeonは通常満たす。公開報告ではSD-Turbo系(1〜4ステップ)がIntel Core i7クラスで1枚0.8秒程度という例があり、通常ステップのCPU量子化推論(1枚10分規模の報告あり)と比べて桁違い。
- 推奨ツール: **FastSDCPU**(rupeshs/fastsdcpu、OpenVINOバックエンド、img2img/denoise制御あり)。Android専用アプリ(Local Dream等)はサンドボックスには不要(Linux上でPythonスタックを直接使える)。
- モデルは§8.2のAnything V5/MeinaMixに**LCM-LoRAを後付け**したものを軸とする(未検証、要小規模テスト)。1990s Anime LCM-1.5等の専用蒸留アニメモデルは画風のミスマッチ(1990年代調)リスクがあるため保留。

### 8.4 まとめ表

| | Pixel 10 Pro | サンドボックス |
|---|---|---|
| 制約 | Tensor NPU非公開でCPU/GPUフォールバック、RAM/ストレージも限定 | CPU律速(RAMは15GBと余裕) |
| 実行環境 | Local Dream(Android) | FastSDCPU(OpenVINO、Python直接) |
| モデル | Anything V5/MeinaMix(通常ステップ) | 同左 + LCM-LoRA(蒸留、速度優先) |
| 未検証事項 | 実機速度・品質(公開情報止まり) | LCM-LoRA適用後の同一性保持、OpenVINO変換の手間 |

### 8.5 サンドボックス実機テスト結果(2026-07-19、ユーザー指示により実施)

`optimum-intel[openvino]` + `diffusers`をvenvに導入し(`experiments/ondevice_gen_test/`)、
実際にサンドボックス上でOpenVINOパイプラインを実行して検証した。

**①動くか**: 動作した。`hsuwill000/LCM-anything-v5-openvino`(HuggingFace公開、
§8.2で採用方針とした「Anything V5 + LCM」の変換済みモデルがそのまま存在)を
`OVStableDiffusionImg2ImgPipeline`でロードし、txt2img/img2imgとも生成できた。

**②速度**: パイプライン初回ロード25〜40秒(モデルDL+OpenVINOコンパイル、以後は
`ov_cache`でキャッシュ)。生成本体は512x512・LCM 2〜4ステップで**1枚あたり6〜7秒**。
公開情報にあった「Core i7クラスで0.8秒/枚」ほどではないが(SDXS-Turboという
より軽量な専用モデルでの数値だったため)、通常ステップCPU量子化推論の
「1枚10分規模」との比較では二桁近く高速。§8.3の想定を裏付けた。

**③キャラクター一貫性**: `characters/ref/front.png`を初期画像としてimg2imgを実施
(結果は`experiments/ondevice_gen_test/out_img2img_s0.3.png`
/`out_img2img_s0.5.png`)。
- **denoise strength 0.3**: ポーズ・ツインテール・プレイド柄セーター・赤スカーフ・
  緑スカート・ブーツの配色がほぼ保持され、目視QCで「同一キャラクターと分かる」
  水準。前髪のリボン等の細部は簡略化された。
- **denoise strength 0.5**: 全体シルエットは分かるが、セーターの柄・色や
  ヘアアクセサリの色が変化し、同一性のブレが目視で明確に出た。

→ 本書§3が既に提案していた「denoise 0.3〜0.4程度」の運用が、実測でも
妥当なレンジであることを確認できた。0.5では同一性ガチャが再発するため
避けるべき。

**残った未検証事項**:
- 今回はSDAI/Local Dream等のAndroidアプリ経由ではなく、サンドボックス上で
  直接`optimum-intel`を叩いた。Pixel実機での同等テストは別途必要。
- 顔まわりの局所差分(目・口だけ差し替え)はまだ試していない。全身img2imgの
  結果のみ。
- 疎格子分解・レイヤー化・Z順メタデータ付与など、後続の実装ステップは未着手。

### 8.6 「可愛くない」フィードバックを受けた追加チューニング(2026-07-19)

§8.5の結果(Anything V5系、`out_img2img_s0.3.png`)はキャラクター一貫性は
保てていたが、目視で「可愛さ」が原画より後退していた(目の輝き・丸みが
控えめ、表情が中立的)。ユーザーから時間をかけて良い方を追い込む許可を
得たため、モデル・プロンプト・パラメータを追加比較した。

**試した変更**:
1. `clip_skip=2`(アニメ系SD1.5モデルの定石設定)を試みたが、
   `hsuwill000/LCM-anything-v5-openvino`のOpenVINO変換済みtext encoderは
   中間層のhidden statesを出力しない変換のされ方で、`IndexError`により
   **使用不可**と判明(再変換すれば使える可能性はあるが、自前のtext encoder
   再エクスポートが要り今回は見送り)。
2. モデルを`iamanaiart/LCM-meinamix_meinaV11-openvino`(MeinaMix)に変更。
3. プロンプトに`kawaii, moe, round soft face, huge sparkling round eyes,
   cheerful smile`等を追加、ネガティブに`mature face, sharp jaw, narrow eyes,
   closed mouth, serious expression`を追加。
4. denoise strengthを0.3→0.2〜0.25、ステップ数を8→10前後に調整。

**結果**: MeinaMix + strength 0.22 + steps 10
(`experiments/ondevice_gen_test/out_meinamix_s022_steps10.png`)が最良。
Anything V5系より目が大きく輝き、口角の上がった笑顔になり、原画の
髪リボン(緑)も保持された。同一性(配色・ポーズ・衣装)は0.3運用時と
同等に保たれている。生成時間は変わらず1枚5〜6秒程度。

**更新した結論**: このキャラクター(丸顔・大きい目のkawaii寄りデザイン)
においては、§8.2で「実績重視」として採用したAnything V5より
**MeinaMixの方が仕上げ工程に適する**。今後は
「MeinaMix + strength 0.2〜0.25 + kawaii系プロンプト」を仕上げチェーンの
既定値とする。§8.2/§8.4の「Anything V5(またはMeinaMix)」は
本節の実測を踏まえ「MeinaMix優先、Anything V5は次点」に読み替える。

### 8.7 品質優先の追加検証: LCM蒸留を外した通常ステップ生成(2026-07-19)

§8.6の結果に対し、ユーザーから「品質を上げたいので時間をかけて良い」との
許可を得たため、さらに2方向を検証した。

**1. `clip_skip=2`の再挑戦(自前エクスポート)**: §8.6でLCM変換済みモデルでは
`clip_skip`が使えないと分かったが、「変換のされ方が悪かっただけでは」という
疑いを検証するため、`Meina/MeinaMix`の生safetensors(V10 baked VAE)を
`StableDiffusionPipeline.from_single_file`→`save_pretrained`→
`optimum-cli export openvino`で**自前で**OpenVINO化した。結果、**同じ
`IndexError`で再現し、使用不可と確定**した。optimum-intelのOpenVINO
text encoderエクスポートが最終層のhidden stateしか出力しない仕様に起因し、
特定の変換元の問題ではないと判明(この制約は本書に確定事項として記録する)。

**2. LCM(蒸留・8〜10ステップ)ではなく通常ステップで生成**: この自前
エクスポートしたMeinaMix V10は蒸留されていない通常モデルなので、
DPM++ 2M Karrasスケジューラ・25ステップ・CFG 7.0で生成できる。結果は
`experiments/ondevice_gen_test/out_v10_full_s030_steps25_cfg7.png`
(strength 0.30)/`out_v10_full_s022_steps25_cfg7.png`(strength 0.22)。

**発見**: 通常25ステップでも**1枚8〜11秒**(LCM 8〜10ステップの5〜6秒と
大差なし)で生成でき、しかも品質は明確に上("line"がきれい、原画の
プレイド柄・配色をより正確に再現、目のハイライトも自然)。本サンドボックスの
CPU・512x512解像度では、UNetが小さくステップ単価が十分軽いため、
「速度優先でLCM蒸留を使う」という前提が成立しない(速度差が誤差程度な
のに品質だけ犠牲になる)ことが分かった。

**更新後の結論**: 仕上げチェーンの既定値を
**「MeinaMix V10(通常版)+ DPM++ 2M Karras + 25ステップ + CFG 7.0 +
strength 0.22〜0.30」**に格上げする。LCM蒸留版(`iamanaiart/LCM-meinamix_meinaV11-openvino`)は
「速度が最優先で品質を妥協してよい場合」の代替に格下げする。

**変換手順(再現用)**:
```
1. huggingface_hub.hf_hub_download("Meina/MeinaMix", "Meina V10 - baked VAE.safetensors")
2. StableDiffusionPipeline.from_single_file(..., torch_dtype=torch.float32, safety_checker=None)
   → .save_pretrained("meinamix_v10_diffusers")
3. optimum-cli export openvino --model meinamix_v10_diffusers --task text-to-image
   --weight-format fp16 meinamix_v10_ov
4. OVStableDiffusionImg2ImgPipeline.from_pretrained("meinamix_v10_ov")
```
変換後のモデル一式(diffusers形式4GB・OpenVINO IR形式2GB)はサイズが
大きいためリポジトリにはコミットしていない(`.gitignore`で除外)。
再現する場合は上記手順を再実行する。

**未検証事項**: Pixel実機(Local Dream)でも同様に通常ステップ版を使うべきか
(Android側の実行速度次第で結論が変わりうる)。他キャラへの一般化。
アップスケール(Real-ESRGAN OpenVINO版`ibrhr/Real-ESRGAN-OpenVINO`を発見
済みだが未検証)。

### 8.8 顔クローズアップ・可愛さ最優先の試作(2026-07-19)

ユーザーから「V11もあるらしい」「重要なパーツは時間をかけてよい」
「原画に忠実でなくていいので可愛さ優先で顔だけ大きく出力してみて」との
指示を受け、追加で検証した。

- **MeinaMix V11**(公式`Meina/MeinaMix_V11`、diffusers形式で配布済み)を
  §8.7と同じ手順(safety_checker除外→`optimum-cli export openvino`)で
  OpenVINO化。
- `characters/ref/front.png`から顔まわりを`(300,20,700,400)`の矩形で
  クロップし768x768にリサイズしたものを初期画像に使用
  (`experiments/ondevice_gen_test/face_crop_raw.png`)。
- 原画忠実度より可愛さを優先する方針に合わせ、denoise strengthを
  **0.55**まで上げ(§8.6〜8.7の0.22〜0.30より大幅に高い、同一性より
  再解釈を許容)、ステップ数も**40**に増やし、`kawaii, extremely cute,
  huge sparkling eyes, rosy cheeks`等を強めたプロンプトで4シード
  (`out_face_cute_seed1〜4.png`)を生成。

**結果**: 目の輝き・頬の赤み・表情の柔らかさが大幅に向上し、明確に
「可愛い」方向へ寄った。ただしdenoiseを上げた分、瞳の色(青→緑になる
シードあり)や背景の意図しないキラキラ演出(seed3)など、原画からの
逸脱も増えた。目視比較では**seed1**(瞳の色が原画に近い青、控えめな
微笑みで破綻が少ない)と**seed4**(緑目だが自然な笑顔)が良好候補。
最終的にどれを採用するかはユーザー確認待ち。

**所要時間の実測**: 768x768・40ステップ・img2imgで**1枚あたり約100秒**。
4シード連続で約7分かかり、ユーザーから「想定より時間がかかった」との
フィードバックを受けた。個別の生成自体は許容された「1分程度」の範囲内
だが、**複数シードを直列でまとめて試す運用は合計時間が伸びる**ため、
今後は1枚ずつ確認しながら少数のシードに絞るか、非同期実行を検討する。

**未検証事項**: この顔クローズアップ手法の他アングル・他キャラへの
一般化。採用するseedの最終決定。

### 8.9 本節の位置づけ

§8.1〜8.4はツール・モデルの比較検討、§8.5は実機テスト結果、§8.6は
「可愛さ」フィードバックを受けた追加チューニング、§8.7は品質優先の
追加検証、§8.8は顔クローズアップ・可愛さ最優先の試作(いずれもユーザー
指示により実施済み)。本格導入(量産・全キャラ展開)はここでは行っておらず、
次のステップ(疎格子分解・表情差分等、および§8.8のseed最終決定)は改めて
ユーザー指示を待つ。

## 9. 関連文書

- [`REAL_ART_LIVE2D_PLAN.md`](./REAL_ART_LIVE2D_PLAN.md) — 全体計画
  (ピクセルパペット方式、VTuber/可愛さ/3Dゲームの実現性)
- [`AI_NATIVE_SUPER_LIVE2D_REVIEW.md`](./AI_NATIVE_SUPER_LIVE2D_REVIEW.md) —
  「スーパーLive2D」としての有効性再検討(ビューリング8方向以上、
  ワープ=補間糊という設計条件)
