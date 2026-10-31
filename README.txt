省メモリ版：5時間・時間変化風速場アプリ

ベース：msm_grib2_5h_temporal_interpolation_fix.zip
変更範囲：MSM-P GRIB2取得・読み込み・キャッシュのみ。5時間計算、毎正時前後30分の適用、FH0/FH3時間線形内挿、モデル式、UIは維持。

省メモリ対策：
- GRIB2ファイルをbytesとしてStreamlitメモリキャッシュに保存しない。1MBずつ一時ディスクにダウンロード。
- GRIB2を1回走査し、必要な気圧面・FH0/FH3・u/v/ghのみデコードする。
- 配列はfloat32、火口周辺を切り出してから時間内挿。
- 風速場のStreamlitキャッシュは最大12件、1時間TTL。
- ダウンロード済みGRIB2は一時ディスクで最大3ファイルまで保持。

既存のvolcanoes.json、CDSEのSecretsを引き継いでください。
依存ライブラリは同梱のrequirements.txtを参照。

注意：構文検証のみ実施。eccodesを使用した実データ読込とStreamlit Cloudでのメモリ計測は未実施。
