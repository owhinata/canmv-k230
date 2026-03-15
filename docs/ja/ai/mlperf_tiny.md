# MLPerf Tiny — K230 DUT 実装

K230 KPU の推論性能を [MLPerf Tiny](https://github.com/mlcommons/tiny) ベンチマークフレームワークで計測するための DUT (Device Under Test) 実装です。

現在は **Image Classification (CIFAR-10, ResNet-8)** に対応しています。

!!! note "K230 と MLPerf Tiny の位置づけ"
    MLPerf Tiny は 10-250MHz / <50mW 級の MCU を典型的な対象としています。K230 はこのカテゴリからは外れますが、submitter API に準拠した DUT を実装することで、公式 harness による標準的な計測手順を再利用できます。

## 前提条件

- K230 SDK がビルド済みであること（ツールチェーン展開済み、MPP ライブラリコンパイル済み）
- SDK がリポジトリルートの `k230_sdk/` に配置されていること
- CMake 3.16 以降
- UART 接続（115200 bps）— MLPerf Tiny legacy harness との通信用

!!! note "SDK のビルド"
    K230 SDK のビルド手順については [SDK ビルド](../development/sdk_build.md) を参照してください。

## 全体ワークフロー

```
[Host PC]                         [K230 DUT]
                                    │
1. git submodule update             │
2. cmake configure/build            │
3. deploy (SCP)                     │
                                    │
4. DUT 起動 (UART)          ──→  main loop
                                    │
5. runner (Python)           ──→  UART コマンド処理
   name%                      ←──  m-name-dut-[...]
   db load N%                 ←──  m-[Expecting N bytes]
   db HEXDATA%                ←──  m-load-done
   infer N W%                 ←──  m-results-[...]
   results%                   ←──  m-results-[...]
```

## ビルド手順

### 1. submodule 取得

```bash
git submodule update --init mlperf_tiny
```

### 2. 設定

```bash
cmake -B build/mlperf_tiny -S apps/mlperf_tiny \
  -DCMAKE_TOOLCHAIN_FILE="$(pwd)/cmake/toolchain-k230-rtsmart.cmake"
```

### 3. ビルド

```bash
cmake --build build/mlperf_tiny
```

### 4. 確認

```bash
file build/mlperf_tiny/mlperf_tiny
```

期待される出力:

```
mlperf_tiny: ELF 64-bit LSB executable, UCB RISC-V, ...
```

## K230 への転送・実行

### deploy ターゲット

kmodel ファイルのパスを指定してデプロイします:

```bash
cmake build/mlperf_tiny -DMLPERF_KMODEL=/path/to/model.kmodel
cmake --build build/mlperf_tiny --target deploy
```

### 手動で転送する場合

```bash
scp build/mlperf_tiny/mlperf_tiny root@<K230_IP>:/sharefs/mlperf_tiny/
scp /path/to/model.kmodel root@<K230_IP>:/sharefs/mlperf_tiny/model.kmodel
```

### K230 bigcore (msh) で実行

```
msh /> /sharefs/mlperf_tiny/mlperf_tiny /sharefs/mlperf_tiny/model.kmodel
```

起動成功時の出力:

```
m-timestamp-mode-performance
m-lap-us-XXXXXXXX
m-init-done
m-ready
```

## UART コマンド手動テスト

minicom 等で bigcore シリアル (`/dev/ttyACM1`, 115200 bps) に接続し、以下のコマンドを送信できます（各コマンドは `%` で終端）:

| コマンド | 説明 | 期待される応答 |
|---------|------|--------------|
| `name%` | デバイス名表示 | `m-name-dut-[unspecified]` |
| `profile%` | プロファイル表示 | `m-profile-[...]` / `m-model-[ic01]` |
| `help%` | ヘルプ表示 | コマンド一覧 |
| `db load 3072%` | 入力バッファ確保 (32×32×3) | `m-[Expecting 3072 bytes]` |
| `infer 1 0%` | 推論 1 回（warmup 0） | `m-results-[...]` |
| `results%` | 最終結果再表示 | `m-results-[...]` |

## Runner による計測

MLPerf Tiny の Python runner を使用して自動計測を行います:

```bash
cd mlperf_tiny/benchmark/runner
python main.py --port /dev/ttyACM1 --baud 115200
```

!!! warning "runner の動作要件"
    runner は legacy UART harness の一部です。MLCommons は新 runner への移行を進めているため、将来バージョンでは手順が変わる可能性があります。

## CMake ターゲット

| ターゲット | コマンド | 説明 |
|-----------|---------|------|
| (デフォルト) | `cmake --build build/mlperf_tiny` | C++ バイナリのビルド |
| `deploy` | `cmake --build build/mlperf_tiny --target deploy` | ビルド + K230 への SCP 転送 |
| `run` | `cmake --build build/mlperf_tiny --target run` | シリアル経由で K230 実行 |

## CMake オプション

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `MLPERF_BENCHMARK` | `ic` | ベンチマーク種別 |
| `MLPERF_KMODEL` | (空) | デプロイする kmodel ファイルのパス |

## ソースファイル

| ファイル | 説明 |
|---------|------|
| `src/main.cc` | エントリポイント — kmodel パス引数、UART メインループ |
| `src/submitter_implemented.cc` | th_* 関数の K230/nncase 実装 |

## トラブルシューティング

### UART 疎通不良

- ボーレートが 115200 bps であることを確認
- bigcore シリアルポート (`/dev/ttyACM1`) を使用していることを確認
- minicom/picocom が占有していないことを確認

### kmodel ロード失敗

- kmodel ファイルのパスが正しいことを確認
- nncase のバージョンと kmodel の互換性を確認

### VB 初期化失敗

- 現在の実装では VB 初期化は省略されています
- nncase runtime が VB を要求する場合は、`submitter_implemented.cc` の `InitPlatform()` に VB 設定を追加してください

## kmodel 変換

`convert_kmodel.py` を使用して、TFLite モデルを K230 KPU 用の kmodel に変換します。変換パイプラインは TFLite → ONNX → kmodel の 2 段階です。

### 依存パッケージのインストール

```bash
pip install tf2onnx tensorflow-cpu onnxsim nncase nncase-kpu
```

### 変換の実行

```bash
python convert_kmodel.py
```

このスクリプトは以下の処理を行います:

1. TFLite モデルを `tf2onnx` で ONNX 形式に変換
2. `onnxsim` で ONNX モデルを最適化
3. `nncase` コンパイラで ONNX を kmodel に変換（K230 KPU 向け）

生成された kmodel ファイルは、前述のデプロイ手順で K230 に転送して使用します。

## ゴールデン推論テスト

`golden_test.py` は、TFLite リファレンス推論と K230 DUT 推論の結果を比較し、モデル変換とデバイス実装の正しさを検証します。

### 実行方法

```bash
python golden_test.py
```

### 動作内容

1. CIFAR-10 テストデータセットから入力画像を取得
2. TFLite インタプリタでリファレンス推論を実行
3. K230 DUT を自動的に起動し、UART 経由で同じ入力を送信
4. DUT の推論結果とリファレンスの推論結果を比較
5. 精度（accuracy）と一致率（agreement）をレポート

DUT の起動・通信は自動化されているため、手動で DUT を起動する必要はありません。

## Runner ベースのベンチマーク

`run_benchmark.py` は、上流の MLPerf Tiny runner を使用して標準的な精度ベンチマークを実行します。

### 実行方法

```bash
python run_benchmark.py
```

### 動作内容

1. CIFAR-10 から IC (Image Classification) 評価データセットを生成
2. 200 サンプルを使用した精度ベンチマークを実行
3. 上流の MLPerf Tiny runner プロトコルに準拠した計測を実施

## 結果サマリ

| 指標 | 結果 | 目標 |
|------|------|------|
| 精度 (accuracy) | 87.5% | 85% |
| レイテンシ | ~2.3ms | — |
| リファレンスとの一致率 | 99% | — |

- 精度は目標の 85% を上回る **87.5%** を達成
- 推論レイテンシは約 **2.3ms** で、K230 KPU の高速推論能力を示しています
- TFLite リファレンスとの一致率は **99%** で、kmodel 変換と DUT 実装の正確さを確認できます
