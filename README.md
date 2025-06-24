# 概要

このスクリプトは、データセットに含まれる文書（ドキュメント）から、構造化されたデータ（エンティティとそれらの関係）を自動的に生成することを目的としています。具体的には、Openrouter の DeepSeek V3 0324 (free) モデルを利用して、非構造化テキストから以下のような情報を抽出・生成し、JSONファイルとして保存します。

1. エンティティ（Entities）: 文書に登場する主要な人物、場所、組織、概念など。
2. 要約（Summary）: 文書全体の要約。
3. 2エンティティ間の関係: 抽出されたエンティティのペア間の関係性。
4. 3エンティティ間の関係: 抽出されたエンティティの3つの組み合わせ（トリプレット）間の関係性。

このようにして生成されたデータは、ナレッジグラフの構築、関係抽出モデルの訓練データ、あるいは高度な質疑応答システムのデータソースなど、様々な用途に利用できます。

# 実行方法

data ディレクトリにマークダウンファイルを配置して、コマンドライン引数を何も指定せずに実行します。
```bash
python entigraph.py
```

実行すると、以下のようなメッセージが表示され、自動で見つかったファイルが順次処理されます。<br>
<br>
No files specified. Searching for markdown files in 'data/*.md'...<br>
Found 3 file(s) to process.<br>
<br>
--- Starting processing for: data/001_Installation.md ---<br>
(処理ログ...)<br>
--- Finished processing for: data/001_Installation.md ---<br>
<br>
--- Starting processing for: data/002_Configuration.md ---<br>
(処理ログ...)<br>
--- Finished processing for: data/002_Configuration.md ---<br>
<br>
<br>
特定のファイルのみを処理する場合は、引数としてファイルパスを指定します。
```bash
python entigraph.py data/001_Installation.md
```

# 生成されるデータセット

各マークダウンファイルから以下のデータセットが生成される。

```
[
  ["エンティティA", "エンティティB", "エンティティC", ...],
  "これはドキュメントの要約です。",
  "エンティティAは、エンティティBを...するために使用されます。",
  "エンティティAとエンティティCは、...という点で競合します。",
  "エンティティA、B、Cは連携して...という機能を実現します。",
  ...
]
```

- 0番目: エンティティ名のリスト ( list[str] )
- 1番目: 文書の要約 ( str )
- 2番目: 各エンティティペア/トリプレット間の関係性を説明する文字列 ( str )
