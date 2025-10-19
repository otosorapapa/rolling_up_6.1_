# デザインシステム指針

ブランドトーン「知性・簡潔・信頼」を体現するためのデザイン・トークンと実装ルールを定義しています。UI開発時は `design_tokens.yaml` および `.streamlit/config.toml` を参照し、コード内で色やフォントをハードコードしないようにしてください。

## カラーパレット

| トークン | 値 | 主な用途 |
| --- | --- | --- |
| `colors.primary` | `#0B1F3B` | 主要アクション、ヘッダー、グラフの主線 |
| `colors.secondary` | `#5A6B7A` | 補助テキスト、境界線 |
| `colors.accent.value` | `#1E88E5` | リンク、タブ選択、強調表示 |
| `colors.accent.emphasis` | `#0B1F3B` | KPI強調、重要ボタンのホバー |
| `colors.accent.soft` | `#56A5EB` | グラデーションやホバー背景 |
| `colors.background` | `#F7F8FA` | アプリ全体の背景 |
| `colors.surface` | `#FFFFFF` | カード・モーダルの基調 |
| `colors.surface_alt` | `#EEF1F5` | テーブルヘッダー、淡い帯域 |
| `colors.text` | `#1A1A1A` | 本文テキスト |
| `colors.muted` | `#4C5A68` | キャプション、補足情報 |
| `colors.success` | `#00A972` | 増加トレンド、成功通知 |
| `colors.warning.value` | `#DFA637` | 注意喚起（彩度を落とした黄橙色） |
| `colors.error` | `#D93025` | エラー、在庫過剰など緊急通知 |

## タイポグラフィ

| トークン | フォントファミリー | サイズ / 行間 | 用途 |
| --- | --- | --- | --- |
| `typography.body` | Inter / Source Sans 3 / Noto Sans JP ほか | 14–16px、行間1.5 | 本文、フォーム、補足説明 |
| `typography.heading` | Inter / Source Sans 3 / Noto Sans JP | 22–28px、行間1.35 | セクション見出し、カードタイトル |
| `typography.numeric` | Roboto Mono / Source Code Pro / Noto Sans Mono CJK | 等幅（タブラー） | KPI値、表の数値列 |

## レイアウト

- `layout.grid.columns = 12`：`st.columns([3,1])` など比率指定で主従関係を明確にします。
- `layout.spacing.unit_px = 8`：余白は 8px グリッドに揃え、16/24/32px 等の倍数を用います。
- `layout.card.radius_px.base = 10`・`shadow = 0 12px 24px rgba(11,31,59,0.08)`：カードは角丸と控えめな影でフラットかつ信頼感を演出します。

## コンポーネントガイド

| コンポーネント | ベース | ルール |
| --- | --- | --- |
| KPIカード | `st.metric()` | 数値・矢印・前期比を表示。`delta_color` で増減を明示し、数値は等幅フォント。 |
| タブ | `st.tabs()` | 売上・粗利・在庫・資金の4タブ構成を基本とし、強調色は `colors.accent` を使用。 |
| セグメントコントロール | `st.radio()` / `st.selectbox()` | 期間・店舗等の切替に使用。ラベルは簡潔にし、並び順で優先度を示す。 |
| トグル | `st.toggle()` | オプション設定。補足説明 (`st.help`) を併記して誤操作を防止。 |
| ツールチップ | `st.help()` | アイコンに説明を集約し、UIを簡潔に保つ。 |

## 実装ポイント

- `.streamlit/config.toml` でテーマカラーとフォントを一元管理しています。変更はこのファイルと `design_tokens.yaml` に反映し、アプリコードでは `core.design_tokens` 経由で取得してください。
- Plotly のカラーパレットは `core.design_tokens.get_plotly_palette()` によりブランドカラーへ自動調整されます。
- Streamlit カスタム CSS では `var(--accent)` などの変数を利用し、値の直接記述を避けます。

これらのガイドラインに沿って UI を実装することで、将来的なブランド変更にも柔軟に対応できる構造になります。
