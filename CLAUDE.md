# CLAUDE.md — canmv-k230

## Git / PR ワークフロー

タスク完了時、指示があればPR作成・更新まで一気通貫で実行する。

- **ブランチ**: `feat/`, `docs/`, `style/`, `fix/`, `build/`, `refactor/`, `chore/` prefix。ベースは常に `main`
- **コミット**: conventional commits 形式 `type: short description`
- **`k230_sdk/`**: サブモジュール。変更はコミットしない（`build_sdk.sh` が実行時にパッチする）

### PR作成

```bash
gh pr create --title "type: short description" --body "$(cat <<'EOF'
## Summary
- 変更点を箇条書き

## Test plan
- [x] テスト項目

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### PR更新

追加コミット & push 後、**必ずPRにコメントを残す**:

```
## type: short description (commit-hash)

変更内容の説明。
```

### PRマージ

```bash
gh pr merge <PR番号> --merge --delete-branch
git remote prune origin
```

## ドキュメント

MkDocs + Material + mkdocs-static-i18n。設定は `mkdocs.yml` 参照。

### 作成手順

1. `docs/ja/` と `docs/en/` に同名 `.md` を作成（日英必須）
2. `mkdocs.yml` の `nav` にエントリ追加（新セクション時は `nav_translations` も）
3. `mkdocs build` で確認

### カテゴリ

- `setup/` — 初期設定、環境構築
- `development/` — サンプルアプリのビルド・実行ガイド
- `technical/` — アーキテクチャ、調査結果

### ソースコード参照

GitHub パーマリンク（コミットハッシュ + 行番号）を使用。ソース変更時はリンクを更新する。

## apps ディレクトリ

各アプリは `apps/<app_name>/` に `CMakeLists.txt` + `src/` で構成。CMake out-of-tree ビルド。

- C: `.c`/`.h`、C++: `.cc`/`.h`
- 新規作成時は既存アプリの `CMakeLists.txt` をテンプレートにする
- コーディングスタイル: Google Style (`clang-format -style=google`)
- cpplint でチェック。SDK 由来で抑制するフィルタ:
  ```
  cpplint --filter=-legal/copyright,-build/include_subdir,-build/namespaces,-build/c++11,-runtime/references,-build/include_order <files>
  ```
  - 上記フィルタ適用後のエラーは **0 件** にすること
  - ヘッダーガードは `APPS_<APP>_SRC_<FILE>_H_` 形式
  - 単一引数コンストラクタには `explicit` を付ける
- `build/` は `.gitignore` で除外済み

### ツールチェーン (`cmake/`)

| ファイル | ターゲット |
|---------|-----------|
| `toolchain-k230-rtsmart.cmake` | bigコア (RT-Smart) — MPP/AI アプリ |
| `toolchain-k230-linux.cmake` | littleコア (Linux) |

### ビルド

```bash
cmake -B apps/<app>/build -S apps/<app> -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-k230-rtsmart.cmake
cmake --build apps/<app>/build
```

## K230 bigcore (RT-Smart msh) の注意点

### シリアル入力の行バッファリング

msh はプログラムの stdin に対して **行バッファリング** を行う。`getchar()` はユーザーが CR (`\r`) を送るまでブロックする。

- プログラムにリアルタイムで文字を渡したい場合、**送信データの末尾に `\r` を付ける**
- プログラム側で `\r` / `\n` が不要な場合（例: MLPerf Tiny の `%` 終端プロトコル）、メインループでフィルタする:
  ```cpp
  if (c == '\r' || c == '\n') continue;
  ```

### printf バッファリング

RT-Smart の `printf` / `vprintf` は出力をバッファする場合がある。シリアル経由で即座にレスポンスを返す必要がある場合は、書き込み後に `fflush(stdout)` を呼ぶ。`setvbuf(stdout, nullptr, _IONBF, 0)` で完全に無効化も可能。

### プロセス終了と msh ハング

bigcore で動作中のプログラムが異常終了すると **msh がハングし、シリアル入力を受け付けなくなる**場合がある。この場合は K230 の **HW リセット**（電源再投入）が必要。

- `Ctrl+C` で停止できる場合もあるが、保証されない
- 長時間実行するプログラムでは SIGINT ハンドラの実装を検討する

### nncase ライブラリのリンク順序

nncase の K230 ランタイムライブラリ (`libnncase.rt_modules.k230.a`) は MPP ライブラリ (`kd_mpi_sys_mmz_*`) に依存する。循環依存を解決するため、MPP と nncase を同じ `--start-group` / `--end-group` ブロックに含める:

```cmake
target_link_libraries(app PRIVATE
    -L${MPP_LIB_DIR}
    -L${_NNCASE_ROOT}/nncase/lib
    -Wl,--start-group ${_MPP_LIBS} -lNncase.Runtime.Native -lnncase.rt_modules.k230 -Wl,--end-group
)
```
