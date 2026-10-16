# Anime Frame Picker

ローカル動画（MP4 / MKV）から、参照画像とシーン条件に合うフレームを
探す FastAPI アプリです。動画・参照画像・抽出画像はこの PC で処理し、
モデルはリポジトリ外へ保存します。

## セットアップ

Windows x64 の Python 3.12 と CUDA 12.8 対応 NVIDIA ドライバーが必要です。
`setup.bat` は `D:\Dev\92_Anime_Frame_Picker\.venv` を作り、Torch
2.9.1+cu128 / TorchVision 0.24.1+cu128 とモデルをインストールします。
モデルの既定保存先は `D:\AI\models\Cold_F\image` です。

```bat
setup.bat
```

通常のセットアップは、メイン解析モデルの準備後にロゴ除去用の
Anime-LaMa 重みも検証します。ロゴモデルは既定で
`D:\AI\models\Cold_F\image\anime-lama\anime-manga-big-lama.pt` に置きます。
既にこの場所に検証済みの重みがある場合はそのファイルを再利用し、古い
仮想環境や別のランタイムへコピーしません。ロゴモデルだけを明示的に準備
する場合は、`.venv` の Python で次を実行します。

```bat
.venv\Scripts\python.exe -m anime_picker.logo_removal prepare
```

メインモデルとロゴモデルを後回しにして、依存関係だけ確認する場合
（`--skip-models` はすべてのモデル準備を省略します）:

```bat
setup.bat --skip-models
.venv\Scripts\python.exe scripts\doctor.py --skip-models
```

`setup.bat --check` は既存環境を確認します。通常起動は `run.bat` で、
ブラウザが `http://127.0.0.1:7866` を開きます。保存先を変える場合は、
セットアップまたは起動前に次を設定してください。
既定の保存先は起動場所に依存しないリポジトリ直下の `outputs/` です。

```bat
set ANIME_PICKER_MODEL_ROOT=E:\models\anime-picker
set ANIME_PICKER_OUTPUT_ROOT=E:\anime-picker-output
```

`source/` に動画と字幕を置けます。入力、モデル、キャッシュ、抽出画像は
Git 管理対象外です。

## 基本操作

1. 先にローカル動画を選びます。`source/` の MP4 / MKV またはファイル選択
   から追加してください。
2. 参照画像を追加するか、動画の時刻を指定して切り出します。参照画像を
   選び、中央の画像上で対象をドラッグすると手動矩形を設定できます。画像を
   クリックすると矩形は全体に戻ります。
3. 条件を `欲しい` / `避けたい` に分け、`対象`（検出した人物・頭部）か
   `フレーム`（画面全体）を選びます。
4. 時間範囲・粗密間隔・候補数を設定して解析し、候補を選んで書き出します。

タグは準備済みモデルの正規語彙と表示された別名だけを受け付けます。
現在の一般条件タグは 15,043 件です。rating や character など他カテゴリは
補助的な証拠であり、検索条件として利用できるとは限りません。未対応語は
警告として返され、別の条件に黙って置き換えられません。対象条件と画面条件を
取り違えると、正しいモデルスコアでも意図と異なる結果になります。

<a id="logo-profile-skill"></a>
## ロゴ除去（書き出し時、既定オン）

ロゴ除去は書き出し時だけに行う Anime-LaMa の AI 補完です。画面の設定は
既定でオンですが、解析のスコア、候補画像、解析プレビューは常に元の画素を
使います。オンで書き出す場合は、元動画の PTS と解像度を保ったフルフレームを
`originals/` に残し、ロゴ除去済みの画像を別に保存します。オフの場合は、選択した
元フレームと画素が一致するPNGを保存します。ロゴ除去は申告した加工範囲
（マスクと境界のなじませ範囲）だけを変更し、元動画の PTS は変更しません。

プロファイルが登録されていないロゴや、登録プロファイルが一致しないフレームは
認識済みとは扱わず、除去をスキップして警告します。新しいロゴは映像ごとに
ローカルでテンプレート、マスク、検出閾値をキャリブレーションする必要があり、
全作品に通用する自動認識は行いません。プロファイルとテンプレートは
`.local/logo-profiles/` に保存し、登録手順は [anime-logo-profile skill](.agents/skills/anime-logo-profile/SKILL.md)
と [ロゴプロファイル仕様](docs/logo-profiles.md) を参照してください。

動画、参照画像、プロファイル、モデル、書き出しはこの PC 内で処理します。
セットアップ時に公開配布元から重みを取得しますが、実行時に動画や画像を
外部推論サービスへ送信しません。ロゴ除去コードは `09_test` のパスや実行時
モジュールに依存しません。

PixAI は対応する CUDA GPU で bfloat16 推論を使います。float32 を明示的に
選ぶ場合は、起動前に `ANIME_PICKER_TAG_DTYPE=float32` を設定してください。
対象cropに複数人物の強い手掛かりがある場合、対象の表情・ポーズ条件の
判定を保留し、候補不足として表示します。

書き出し画像は元動画の該当 PTS を正確に再デコードし、元の解像度で保存します。
アップスケールは行いません。既定のリポジトリ直下 `outputs/` の下に動画名・時刻・
一意 ID を含む実行フォルダを作り、`metadata.json` に候補ごとの PTS、時刻、
timebase、矩形、ロゴ除去の適用状態などを保存します。

これは候補を順位付けする補助機能で、すべての絵柄、部分的に見える目、解剖学的
な正しさを保証するものではありません。まず短い代表区間で参照矩形と条件を
確認してから、探索範囲を広げてください。

## 開発確認

```bat
.venv\Scripts\python.exe scripts\doctor.py --skip-models
.venv\Scripts\python.exe -m pytest
```

`doctor.py` は依存関係、Torch CUDA、実際の ONNX Runtime CUDA セッションを
確認し、CPU 版と GPU 版の ONNX Runtime が同居している場合は失敗します。
通常の診断ではメインモデルに加えてロゴモデルと登録プロファイルの準備状態も
表示します。ロゴモデルが無くても、ロゴ除去をオフにすればメイン解析は利用できます。
固定済み依存関係は [requirements.lock.txt](requirements.lock.txt)、モデルの
版とライセンスは [docs/model-licenses.md](docs/model-licenses.md) を参照してください。
