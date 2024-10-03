#概要
以下のフローにより動画データから議事録を作成するアプリです。
①動画データ(mp4)をアップロード
②音声データに切り分け
③文字起こし
④話者分離
⑤文章の綺麗化及び表形式による出力

#実施ステップ
・cloneした後、仮想環境を作成頂き、requirements.txtの内容を反映頂く
・.envファイルを作成し、OPENAIのAPIkeyとHuggingFaceのAPItokenを定義頂く
・管理者権限でcmdを実行した後、streamlit run app.pyにてアプリを起動する
