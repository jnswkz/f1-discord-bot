# Simple F1 Discord Bot

Formula 1（F1）の最新情報を Discord で手軽に確認できる、シンプルなプレフィックス型ボットです。  
このプロジェクトは Python 製で、パッケージ管理・実行には「uv」を使用します（Poetry は不要）。

> 注: 実装の詳細（ファイル名・API など）はコードに合わせて適宜読み替えてください。

---

## 目次

- [主な機能](#主な機能)
- [コマンド一覧](#コマンド一覧)
- [必要要件](#必要要件)
- [セットアップ（uv）](#セットアップuv)
- [環境変数](#環境変数)
- [実行方法](#実行方法)
- [トラブルシューティング](#トラブルシューティング)
- [ライセンス / 謝辞](#ライセンス--謝辞)

---

## 主な機能

- F1 ニュースの取得（要件に応じたニュースソース）
- 直近/次戦のセッション情報（FP/予選/決勝など）
- 年度別のドライバーズ選手権（WDC）順位
- 年度別のコンストラクターズ選手権（WCC）順位

---

## コマンド一覧

デフォルトのプレフィックスは「$」です。

- `$news`  
  最新の F1 ニュースを表示します。

- `$session`  
  直近または次戦のセッションスケジュールを表示します。

- `$wdc {year}`  
  指定年のドライバーズ選手権順位を表示します。  
  例: `$wdc 2025`

- `$wcc {year}`  
  指定年のコンストラクターズ選手権順位を表示します。  
  例: `$wcc 2025`

> 年の省略可否やデフォルト年は実装に合わせて調整してください。

---

## 必要要件

- Python 3.10 以上推奨
- Discord Bot トークン（[Discord Developer Portal](https://discord.com/developers/applications) から取得）
  - MESSAGE CONTENT INTENT（メッセージ内容の特権インテント）を有効化してください（プレフィックスコマンドのため）

---

## セットアップ（uv）

1) リポジトリ取得
```bash
git clone https://github.com/jnswkz/simple-f1-discord-bot.git
cd simple-f1-discord-bot
```

2) uv のインストール（未導入の場合）
- macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- Windows（PowerShell）:
```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3) 仮想環境と依存関係
- uv でインストールできます:
```bash
uv sync
```

4) .env を作成（後述の「環境変数」を参照）

---

## 環境変数

プロジェクト直下に `.env`（またはホスティング環境の環境変数）を設定します。

```dotenv
# 必須
TOKEN=あなたのDiscordBotトークン

NEWS_CHANNEL_ID=あなたのF1 ニュース Channel
```

---

## 実行方法

エントリーポイントのファイル名に合わせて実行してください（例: `main.py` や `bot.py`）。

- uv で実行:
```bash
uv run python main.py
# または
uv run python bot.py
```

- 通常の実行（仮想環境を有効化済みの場合）:
```bash
python main.py
```

Bot をサーバーへ招待し、必要な権限（特に MESSAGE CONTENT INTENT）を満たしていることを確認してください。

---

## トラブルシューティング

- コマンドに反応しない  
  - Bot がオンラインか、対象サーバーへ招待済みか確認  
  - Developer Portal で MESSAGE CONTENT INTENT が有効か確認  
  - トークンや権限（メッセージ閲覧・送信権限）が正しいか確認

- 時刻表示がズレる  
  - 取得元 API が UTC の場合があります。表示時にタイムゾーン変換を実装してください。

---

## ライセンス / 謝辞

- ライセンス: プロジェクトに合わせて記載してください
- Discord ライブラリ（discord.py）に感謝します



