## こいしの日2024： ペルソナ主導ロールプレイにおけるグローバル忠実度の定量化と最適化

- 幻想を現に

## 導入

パーソナ主導型ロールプレイ（Persona-driven Role-playing, PRP）は、数段落の短い文章でAIキャラクターを構築できるほど素晴らしいものですが、すべての人設声明に忠実にAIキャラクターを保つことは難しい問題です。PRPエージェントは、常に多くの間違いを犯すか、知るべき知識についても常に曖昧です。

![Case](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/case_koishi.png)

この制限の主な理由は、グローバルなPRP忠実度を定量化できる指標が欠けているためです。そのため、我々は人間の直感に従って以下の指標を提出しました：

![APC](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/apc_koishi.png)

簡単に言うと、ユーザーからのクエリが来るたびに、各人設声明はアクティブ（クエリに関連する）またはパッシブ（クエリに無関係）な制約になります。アクティブな制約を満たすためには、応答は声明によって導かれる必要があります（声明の情報を含む必要があります）。一方、パッシブな制約については、応答はそれらに矛盾しない内容であれば良いのです（人設声明によれば間違った情報を含まない内容であれば良い）。

![DPO](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/dpo_koishi.png)

我々はすべての人設声明を検証し、その制約が満たされているかどうかを確認します。満たされた制約の数をカウントし、それをグローバルなPRP忠実度を評価する指標として使用します。この指標はアクティブ・パッシブ・コンストレイント（APC）スコアと名付けられています。ダイレクト・プリファレンス・オプティマイゼーション（Direct Preference Optimization, DPO）は、モデルが人間や基準によって好まれる応答のように生成することにリワードを与える方法です。したがって、同じクエリに対して二つの応答をサンプリングし、それらのAPCスコアに基づいてDPOを適用することで、PRPエージェントが人設声明に対してよりグローバルに忠実であるように導くことができます。

応用の実現は、APCスコアは確率的モデルによって割り当てられ、より正確な推定を目指します。この推定は、声明とクエリの関連性の確率および声明から応答への自然言語推論の確率として、以下のように形式化されます。

![Formula](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/apc_formula.png)

数式が気にいらない場合は、最小限の知っておくべきことは、**関連性**と**自然言語推論**のために2つの確率的推定器が必要であるということです。

![Distillation](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024/blob/main/images/distillation_koishi.png)


したがって、上記のパイプラインを使用して、合成データセットからGPT-4を精製することにより、そのような推定器を構築します。これまでのところ、グローバルなPRP忠実度の定量化と最適化のパズルは完成しています。では、忠実なPRPエージェントを構築する旅にお祝いを！
