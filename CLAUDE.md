# CLAUDE.md

このリポジトリで作業するAI/開発者向けのガイドです。編集前に必ず目を通してください。

## プロジェクト概要

VTuberアセットエンジンの検討・実装リポジトリ。詳細な設計・経緯は
[`VTUBER_ASSET_ENGINE_PLAN.md`](./VTUBER_ASSET_ENGINE_PLAN.md)を参照。
外部AI・GPUを使わずスマホ+Claudeのみで完結する「ベクターパペット」方式
(手法書: [`SUPER_LIVE2D_MOBILE_METHOD.md`](./SUPER_LIVE2D_MOBILE_METHOD.md)、
PoC: `poc/vector_puppet_poc.html`)を実証済み。それ以外は計画段階。

## 前身リポジトリとの関係

このプロジェクトは[`amanesf/ghostitd`](https://github.com/amanesf/ghostitd)
での動画→3D化(visual hull)の検討から派生した。以下はその関係で押さえて
おくべき点:

- `ghostitd`は別プロジェクト(3Dtool: `landmark_tool.html` /
  `character_3d.html` / `controller.html` / `js/`パイプライン)であり、
  このリポジトリからは独立して開発が続く。**コードを混在させない。**
- `VTUBER_ASSET_ENGINE_PLAN.md`内の相互参照(`js/accessories.js`の
  アクセサリー体系、`VEO_VIDEO_TO_3D_PLAN.md`の座標系など)は`ghostitd`
  側のファイルを指しており、必要なら`ghostitd`をクローンして参照する。
- 計画書のパーツタクソノミー設計(§3)は`ghostitd`の
  `js/accessories.js`・`js/common.js`の`LM_GROUP_ORDER`を土台にする方針
  なので、ゼロから設計し直さないこと。

## 次に着手しうる作業(計画書「未決着・次回への引き継ぎ」より)

1. パーツ/表情の標準マニフェスト(JSON)の設計
2. 角度ガイド生成ツール(Three.jsプロキシメッシュ→N方向レンダリング)
3. ローカル画像生成AI環境構築+小規模テスト生成
   (ユーザーからの明示的な着手指示待ち。サンドボックスにGPUが無いことを
   計画書§5で確認済み)

## 実イラストをそのまま動かす方式(現行方針、着手はユーザー指示待ち)

`characters/blonde_twintails.html`(ベクター再描画)は原画と画風が乖離し
不採用。現行方針は**原画ピクセルをレイヤー分解して動かす「ピクセル
パペット」方式**。統合計画は
[`REAL_ART_LIVE2D_PLAN.md`](./REAL_ART_LIVE2D_PLAN.md)(v2、**これが現行
マスタープラン**)。要点:

- 設計原則: **可愛さは常に2D、3Dは裏方**/実絵キーフレーム主役・ワープは
  補間糊([`AI_NATIVE_SUPER_LIVE2D_REVIEW.md`](./AI_NATIVE_SUPER_LIVE2D_REVIEW.md))
  /全体再生成禁止/疎格子(フル分解は正面±45°のみ)
- ビューリング生成の主役は**CharacterGen**(アニメ特化・Aポーズ正規化)、
  控えHunyuan3D(裏方)、保険Zero123(本キャラで画風保持実証済み)、
  局所差分はGemini、仕上げはPixel 10 Pro
  ([`ON_DEVICE_GENERATION_PLAN.md`](./ON_DEVICE_GENERATION_PLAN.md)は調査記録)
- 着手順の先頭は「正面のみVTuberパペット(MVP)」— 生成AI不要で
  既存front.pngから作れる
- 画像生成はサンドボックスで実証済み(MeinaMix V10/V11+OpenVINO、
  `ON_DEVICE_GENERATION_PLAN.md`§8)。エンジン本体の設計目標は
  [`PUPPET_ENGINE_DESIGN.md`](./PUPPET_ENGINE_DESIGN.md)(表情・アクション・
  ダンス・衣装替え・視線・会話+音声AI・ビューア。タッチ反応は不採用)
- 原画4アングルは`characters/ref/`に到着済み。要件はVTuber/可愛さ維持/
  3Dゲーム(キャラはビルボードスプライトが正式主役)の3つ
**ユーザーから「勝手に実装しない」指示あり — 実装着手は明示指示を待つこと。**

## 現在の実装状況(2026-07-20)

実装は着手済み(ユーザー指示)。現在地は
[`PUPPET_ENGINE_DESIGN.md`](./PUPPET_ENGINE_DESIGN.md)のv3改訂冒頭を参照:

- エンジン(旧世代): `puppet/viewer.html`(グリッドメッシュワープ+2次ばね駆動、
  silver_ponytailキャラ向け)。単一ファイル版`puppet/viewer_standalone.html`は
  GitHub Pagesで配信(https://amanesf.github.io/new2D3D/ 、リポジトリ名は
  大文字小文字を区別するため`new2d3d`小文字だと404になる点に注意)。
  この方式は「ワープが主役」になりがちで歪み・レイヤー境界の不連続が
  既知の弱点(AI_NATIVE_SUPER_LIVE2D_REVIEW.md参照)。解消は
  パーツ別デフォーマ化+ワープを補間糊に格下げする設計転換で対応する方針
- キャラ: ゼロベース再生成の`characters/zero/master_bust.png`が現行マスター
  (旧silver_ponytail資産はpuppet/layers等に残るが旧世代)
- 表情/viseme: `characters/zero/expr_adopted/`にVRM標準5表情+neutral、
  viseme5形状(aa/ih/ou/ee/oh)を採用確定済み。生成スクリプトは
  `puppet/gen_expr.py`・`puppet/gen_viseme.py`(顔内側フェザー領域を
  まるごと差し替える方式。目・口の独立パッチ化はしない)
- 精密レイヤー分解(顔領域スコープ): `puppet/build_face_layer.py`で
  `master_bust.png`をbase(体)/head(髪・耳・ヘッドセット)/
  face_feather_mask(表情差し替え穴)に分解済み(`puppet/layers_zero/`)。
  体・ポニーテールの分解は未着手
- 新ビューア: `puppet/viewer_zero.html`で表情/viseme差し替えを実機確認済み
  (Playwrightでの描画テスト実施、継ぎ目なく合成できることを確認)
- 生成環境: OpenVINO+MeinaMix V11(モデルはgit管理外。
  `gen_master.py`等の手順で再構築。本サンドボックスでは
  torch/diffusers/optimum-intel/openvino一式もpipで都度再インストールが必要)
- 次工程: **[`SUPER_LIVE2D_V3_PLAN.md`](./SUPER_LIVE2D_V3_PLAN.md)(2026-07-21、
  AIネイティブ全体再設計。これが現行マスタープラン)**に従う。パーツ別
  メッシュ変形(WebGL)+パラメータシステム+自動リグ+自動継ぎ目QC+
  ワークフロー化(2キャラ目コスト表が受け入れ基準)。実装は2026-07-21
  着手済み(同計画書改訂1)。マスターは**ユーザー提供原画の生成AI変換版**
  `characters/zero/master_v2_fullbody.png`が正(旧master_bust系譜は旧世代)。
  検証は同計画書の検証プロトコル(全景判断禁止・反証駆動・証拠画像つき
  報告)に必ず従うこと。
- **頭部360度ビューリングの実現性、側面1角度で好転(2026-07-21、
  同計画書改訂3)**: 「スーパー」の名分である側面・背面込み全周ビュー
  (Live2Dが原理的に不可能な領域)を、頭部の疑似3D・360度回転・
  アクションの生成AI路線(離散キーフレーム生成、改訂2)で実現できるか
  検証中。ControlNet頭部回転・Zero123++(CPU)は側面・背面で不合格
  (同一性崩壊/平面イラストがカード状に破綻)、CharacterGen公開デモは
  機能停止で未検証、クラウドGPU課金はPixel実機のみという制約から見送り
  だったが、**IP-Adapter+ControlNet+MeinaMix(ステップ数40・
  77トークン制限対応版)で側面90°の識別要素(髪色・瞳・ヘッドセット・
  衣装)を初めて実用域で再現できた**(§6実現性確認の結果⑧)。ただし
  検証は側面1角度・1シードのみで、背面・複数シードでの再現性は未確認。
  次工程はこの角度を広げる検証(背面・複数シード)が最優先
  (QCゲート実装・SAMスパイクS3等の他工程より先)
  前段の品質向上計画は[`VIEWER_QUALITY_PLAN.md`](./VIEWER_QUALITY_PLAN.md)
  (2026-07-20レビュー起点)。要点: 矩形切り+穴埋め方式が「四角い
  枠・切れ目」の構造的原因と診断済み → base無傷+輪郭アルファ抽出+縁
  フェザーの「Live2D式重ね」へ転換し、まばたき・口パク自動・視線・
  ポニーテール多段振り子・ポインタ追従・アクションプリセット・アイドル
  表情演出・後れ毛スウェイまで実装済み(同計画書§1 C・§2)
- **設計上の反省**(ユーザー指摘、2026-07-20): ポニーテールの多段振り子は
  実装当初、切り出しポリゴンの座標を誤って右腕を掴んでおり、「ポニーテール
  のつもりが腕が動く」不具合になっていた(修正済み)。また後れ毛は最初の
  レイヤー分解(build_face_layer.py)の時点で「動かす前提」で房を分離して
  いなかったため、後から独立房として切り出そうとして境界があいまいで
  苦労した(結局、矩形+ガウスぼかしの柔らかいマスク方式で解決)。
  **教訓: 次にキャラのレイヤー分解を最初から設計するときは、パーツを
  切り出してから動かし方を考えるのではなく、どう動かしたいか(独立して
  揺らしたい房はどれか)を先に決めてから分解境界を引くこと。**
