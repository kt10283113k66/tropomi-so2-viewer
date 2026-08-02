# TROPOMI SO₂ Viewer — Streamlit公開版

Sentinel-5P/TROPOMIのSO₂分布と、京都大学生存圏研究所で公開されている
気象庁MSM-Pを用いた準定常ガス拡散モデルを地図上で比較するStreamlitアプリです。

## 公開版の特徴

- GitHubからStreamlit Community Cloudへ直接デプロイ可能
- CDSE認証情報をStreamlit Secretsで管理
- 任意の簡易パスワード認証
- TROPOMI SO₂、雲量、ピーク位置の表示
- MSM-Pの複数時刻・気圧面から最適モデルを選択
- モデル平均化分解能を任意設定
- 推定放出率・補正放出率を表示
- Pythonの詳細なエラー内容を一般利用者に表示しない設定

## ローカルでの起動

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`.streamlit/secrets.toml`を作成します。

```toml
CDSE_CLIENT_ID = "実際のClient ID"
CDSE_CLIENT_SECRET = "実際のClient Secret"
APP_PASSWORD = "任意の公開用パスワード"
```

起動します。

```cmd
streamlit run app.py
```

## GitHubへのアップロード

1. GitHubで新しいリポジトリを作成します。
2. このフォルダ内のファイルをすべてリポジトリへアップロードします。
3. `.streamlit/secrets.toml`はアップロードしません。
4. `secrets.example.toml`にはダミー値だけを残します。

Gitコマンドを使う場合:

```cmd
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

## Streamlit Community Cloudへの公開

1. GitHubアカウントでStreamlit Community Cloudへサインインします。
2. `Create app`を選択します。
3. Repository、Branch (`main`)、Main file path (`app.py`)を指定します。
4. App settingsのSecretsへ以下を貼り付けます。

```toml
CDSE_CLIENT_ID = "実際のClient ID"
CDSE_CLIENT_SECRET = "実際のClient Secret"
APP_PASSWORD = "公開用パスワード"
```

5. Deployを押します。
6. `https://任意の名前.streamlit.app`形式のURLが発行されます。

## 公開時の注意

- CDSEのClient SecretをGitHubへコミットしないでください。
- `APP_PASSWORD`を削除すると、URLを知る人が誰でも利用できます。
- 複数人が同時に解析すると外部API、メモリ、通信量の負荷が増えます。
- 本アプリの結果は簡略化モデルによる参考値であり、公式な火山監視情報ではありません。
- Community Cloudでは長時間処理や大容量データ処理に制約が生じる場合があります。
