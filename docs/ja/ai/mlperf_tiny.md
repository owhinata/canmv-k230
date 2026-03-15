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
