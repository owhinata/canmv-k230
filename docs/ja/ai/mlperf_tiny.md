# MLPerf Tiny — K230 DUT 実装

K230 KPU の推論性能を [MLPerf Tiny](https://github.com/mlcommons/tiny) ベンチマークフレームワークで計測するための DUT (Device Under Test) 実装です。

**IC (Image Classification)、VWW (Visual Wake Words)、KWS (Keyword Spotting)、AD (Anomaly Detection)** の 4 ベンチマークに対応しています。1 つのバイナリ + ベンチマーク別 kmodel で全ワークロードを実行可能です。

!!! note "K230 と MLPerf Tiny の位置づけ"
    MLPerf Tiny は 10-250MHz / <50mW 級の MCU を典型的な対象としています。K230 はこのカテゴリからは外れますが、submitter API に準拠した DUT を実装することで、公式 harness による標準的な計測手順を再利用できます。

## 対応ベンチマーク

| ID | タスク | モデル | 入力サイズ | 出力 | 精度目標 |
|----|--------|--------|-----------|------|---------|
| ic01 | Image Classification | ResNet-8 | 32×32×3 = 3,072 | 10 classes | 85% |
| vww01 | Visual Wake Words | MobileNetV1 | 96×96×3 = 27,648 | 2 classes | 80% |
| kws01 | Keyword Spotting | DS-CNN | 49×10×1 = 490 | 12 classes | 90% |
| ad01 | Anomaly Detection | AutoEncoder | 1×640 | 640 (再構成) | AUC 0.85 |

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
3. deploy (binary + kmodels)        │
                                    │
4. DUT 起動 (UART)          ──→  main loop (auto-detect benchmark)
                                    │
5. runner / golden_test     ──→  UART コマンド処理
   name%                      ←──  m-name-dut-[...]
   db load N%                 ←──  m-[Expecting N bytes]
   db HEXDATA%                ←──  m-load-done
   infer N W%                 ←──  m-results-[...]
```

## ビルド手順

### 1. submodule 取得

```bash
git submodule update --init mlperf_tiny
```

### 2. 設定・ビルド

```bash
cmake -B build/mlperf_tiny -S apps/mlperf_tiny \
  -DCMAKE_TOOLCHAIN_FILE="$(pwd)/cmake/toolchain-k230-rtsmart.cmake"
cmake --build build/mlperf_tiny
```

### 3. kmodel 変換

全ベンチマークの kmodel を一括生成:

```bash
cmake --build build/mlperf_tiny --target kmodel
```

個別に変換する場合:

```bash
.venv/bin/python apps/mlperf_tiny/scripts/convert_kmodel.py --benchmark ic01
.venv/bin/python apps/mlperf_tiny/scripts/convert_kmodel.py --benchmark ic01 vww01 kws01 ad01
```

### 4. デプロイ

バイナリ + 全 kmodel を K230 に転送:

```bash
cmake --build build/mlperf_tiny --target deploy
```

## K230 上の配置と実行

```
/sharefs/mlperf_tiny/
  mlperf_tiny          # 共通バイナリ
  ic01.kmodel
  vww01.kmodel
  kws01.kmodel
  ad01.kmodel
```

### 実行

```
msh /> /sharefs/mlperf_tiny/mlperf_tiny /sharefs/mlperf_tiny/ic01.kmodel
msh /> /sharefs/mlperf_tiny/mlperf_tiny /sharefs/mlperf_tiny/vww01.kmodel
msh /> /sharefs/mlperf_tiny/mlperf_tiny /sharefs/mlperf_tiny/ad01.kmodel
```

model_version は kmodel ファイル名から自動検出されます。明示指定も可能:

```
msh /> /sharefs/mlperf_tiny/mlperf_tiny /sharefs/mlperf_tiny/ad01.kmodel ad01
```

## テスト (ctest)

```bash
ctest --test-dir build/mlperf_tiny -R ic01$         # IC benchmark のみ
ctest --test-dir build/mlperf_tiny -R ic01_golden   # IC golden test のみ
ctest --test-dir build/mlperf_tiny -R ic01          # IC benchmark + golden
ctest --test-dir build/mlperf_tiny -E golden        # 全ベンチマーク benchmark（golden 除外）
ctest --test-dir build/mlperf_tiny -R golden        # 全ベンチマーク golden test
ctest --test-dir build/mlperf_tiny                  # 全テスト
```

## ゴールデン推論テスト

`golden_test.py` は、TFLite リファレンス推論と K230 DUT 推論の結果を比較し、モデル変換とデバイス実装の正しさを検証します。

```bash
.venv/bin/python apps/mlperf_tiny/scripts/golden_test.py --benchmark ic01 -n 100
.venv/bin/python apps/mlperf_tiny/scripts/golden_test.py --benchmark vww01 -n 10
.venv/bin/python apps/mlperf_tiny/scripts/golden_test.py --benchmark kws01 -n 10
.venv/bin/python apps/mlperf_tiny/scripts/golden_test.py --benchmark ad01 -n 10
```

## Runner ベースのベンチマーク

`run_benchmark.py` は、上流の MLPerf Tiny runner を使用して標準的なベンチマークを実行します。

```bash
.venv/bin/python apps/mlperf_tiny/scripts/run_benchmark.py --benchmark ic01
.venv/bin/python apps/mlperf_tiny/scripts/run_benchmark.py --benchmark vww01 --mode p
```

## データセット

評価データセットは自動ダウンロード・キャッシュされます:

| ベンチマーク | ソース | キャッシュ | サンプル数 |
|------------|--------|-----------|-----------|
| ic01 | CIFAR-10 (Keras) | `~/.keras/datasets/` | 200 |
| vww01 | COCO2014 96×96 (Silicon Labs) | `~/.mlperf/vww/` | 1,000 |
| kws01 | Speech Commands v2 (tfds) | `~/.mlperf/kws/` | 1,000 |
| ad01 | ToyADMOS/ToyCar (Zenodo) | `~/.mlperf/ad/` | 248 |

## 結果サマリ

| ベンチマーク | 精度 | 目標 | レイテンシ | kmodel |
|------------|------|------|-----------|--------|
| IC (ic01) | 87.5% | 85% | ~2.4 ms | 88 KB |
| VWW (vww01) | — | 80% | ~2.7 ms | 286 KB |
| KWS (kws01) | — | 90% | ~2.4 ms | 39 KB |
| AD (ad01) | — | AUC 0.85 | ~2.5 ms | 315 KB |

- IC は目標の 85% を上回る **87.5%** を達成
- 全ベンチマークのレイテンシは **2.4〜2.7 ms** で、入力サイズに依らず安定
- VWW/KWS/AD の精度は runner による評価データセット全体での計測が必要

## ソースファイル

| ファイル | 説明 |
|---------|------|
| `src/main.cc` | エントリポイント — kmodel パス・model_version 引数、UART メインループ |
| `src/submitter_implemented.cc` | th_* 関数の K230/nncase 実装（全ベンチマーク共通） |
| `src/internally_implemented.cpp` | submodule からコピー、TH_MODEL_VERSION をランタイム変数化 |
| `scripts/convert_kmodel.py` | TFLite → kmodel 変換（全ベンチマーク対応） |
| `scripts/run_benchmark.py` | Runner ベースのベンチマーク実行 |
| `scripts/golden_test.py` | TFLite vs DUT ゴールデンテスト |

## トラブルシューティング

### UART 疎通不良

- ボーレートが 115200 bps であることを確認
- bigcore シリアルポート (`/dev/ttyACM1`) を使用していることを確認
- minicom/picocom が占有していないことを確認

### kmodel ロード失敗

- kmodel ファイルのパスが正しいことを確認
- nncase のバージョンと kmodel の互換性を確認

### VWW のデータ転送が遅い

VWW は入力サイズが 27,648 bytes と大きく、115200 baud のシリアル転送では 1 サンプルあたり約 30-60 秒かかります。これは UART プロトコルの制約であり、推論自体は ~2.7 ms です。
