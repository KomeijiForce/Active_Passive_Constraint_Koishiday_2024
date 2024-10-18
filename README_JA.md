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

## 事前準備

旅立つ前に、以下の事項を整えるのが必要です。

この[ページ](https://docs.anaconda.com/free/miniconda/miniconda-install/)の指示に従ってMiniConda3をインストールしましょう。

この[ページ](https://openai.com/index/openai-api/)の指示に従ってOpenAIアカウントを作成してOpenAI APIを申し込みましょう。

この[ページ](https://huggingface.co/settings/tokens)の指示に従ってHuggingfaceアカウントを作成して読み込みTokenを申し込みましょう。

このリポはGemma-1.1-7b-itに基づいて実現しました、この[ページ](https://huggingface.co/google/gemma-1.1-7b-it)の指示に従ってGemmaモデルへのアクセスを申し込むことが必要な点にご注意ください。

## ファストスタート

### APCに基づいたDPOでRAGモデルを学習させる

最も忠実的なパーソナ主導型ロールプレイエージェントの学習シナリオを簡単なbash命令にしました。 ```bash_is_all_you_need.sh```中の```openai_key``` と ```hf_token```は自分のトーケンに入れ替えて、そして
```bash
bash bash_is_all_you_need.sh
```

このコマンドはAliceに（```wiki```にもっと詳しく）APCに基づいたDPOでRAGシステムを作り出します。そして```prp_models```フォルダにPRPエージェントのLoRAパラメーターが保存されて、```statement```フォルダに生成されたデータセットも保存されて、```discriminators```フォルダにディスクリミネーターも保存されました（```use_pretrained_discriminator```をFalseにした場合）。

好きなキャラクターの高度なPRPシステムを構築するには、wikiテキスト（段落は```\n\n```で区切ります）を```wiki```フォルダに配置し、ファイル名を {character_name}_wiki.txt としてください。その後、```bash_is_all_you_need.sh```内の character を該当するキャラクターに入れ換え、スクリプトを実行します。必要なものは、対応するディレクトリに全て生成されます。

GPUの最適化
GPUの利用は最適化されていますが、このbashコマンドを実行するには40GB以上のGPUメモリが必要です。

推奨パラメータ
```model_engine```: "gpt-4"
提示（プロンプト）はGPT-4専用に設計されています。他のLLMを使用するとエラーが発生する可能性があります。

```use_pretrained_discriminator```: True
通常、有効にして生成の関連性とNLI（自然言語推論）データセットのコストを削減します。（それでもキャラクターのステートメントとユーザーからのクエリの生成は必要です。）

```prp_scale```: "7b"、"2b"
Gemmaモデルは、役割の演じ分けを常に拒否するため、こちらのスケールを使用します。

```max_dpo_data```: 100
約100個のキャラクターステートメントでDPOデータセットを構築する場合、通常1時間以内に完了します。

```lora_rank```: 32以上
LoRAランクが低すぎると、キャラクターの演じ分け性能が低下します。

```rag_top_k```: 4～6
分析の結果、この範囲で最良の性能が得られることが示されています。

APCスコアでリスポンスを評価する
```score.py```ファイルには```classifier.py```中に学習されたディスクリミネーターはAPCスコアを実現することが可能です。```score_APC```ファンクションに基づいて、すべてのペルソナステートメントによるエージェントの忠実度が判明できます，```evaluation_example.py```に以下の通りにユースケースも提供されました。
```python
from classifier import Classifier, get_relevance_discriminator, get_nli_discriminator
from score import score_apc, score_APC

relevance_discriminator = get_relevance_discriminator(character=None, statement_query_relevance_dataset=None, relevance_finetune_epoch=None, use_pretrained_discriminator=True)
nli_discriminator = get_nli_discriminator(character=None, statement_to_response_nli_v2_dataset=None, nli_finetune_epoch=None, use_pretrained_discriminator=True)

character = "Komeiji Koishi"
statements = ["Komeiji Koishi lives with her sister, Komeiji Satori.", "Komeiji Koishi lives in Chireiden."]
query = "Where do you live, Koishi?"
responses = ["I live in Chireiden with my sister, Satori!", "I live in Chireiden!", "I live in Hakurei Shrine!"]
print([score_APC(character, statements, query, response, relevance_discriminator, nli_discriminator).item() for response in responses])

# [1.6079180240631104, 0.9955980777740479, 0.03315635025501251]
```

以上にスコアによって，APCスコアはどうPRPの忠実度を反応するのは初歩的な印象を残します。

### AIキャラとチャットしましょう！

APCに基づいたDPOを実行した後、あなたのキャラクターのLoRAウェイトは```prp_models/gemma-1.1-7b-it-lora-{character}-rag-dpo```に保存されます。これを使用してAIキャラクターとのチャットが可能になります。```chat_example.py```でサンプルも以下の通りに提供されています。

```python
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prp_model import retrieval_augmented_generate
from classifier import get_relevance_discriminator

character = "Your Character"

statements = [data["statement"] for data in json.load(open(f"statement/{character}.json"))]

model_id = f"prp_models/gemma-1.1-7b-it-lora-{character}-rag-dpo"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

prp_tokenizer = AutoTokenizer.from_pretrained(model_id)
prp_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

relevance_discriminator = get_relevance_discriminator(character=None, statement_query_relevance_dataset=None, relevance_finetune_epoch=None, use_pretrained_discriminator=True)

print(f"You are chatting with {character}!")

with torch.no_grad():
    
    while True:
    
        _, response = retrieval_augmented_generate(character, statements, input('User: '), prp_model, prp_tokenizer, relevance_discriminator, rag_top_k=5)
        response = character+": "+response.replace("<eos>", "")
        print(response)
```

以下の例は古明地こいしとのチャット:

```
User: Hi, Koishi! What is your ability?
Komeiji Koishi: I call it the "Silent Whisperer." It allows me to manipulate the unconsciousness of others, making me invisible and granting me control over their actions.
User: Where do you live?
Komeiji Koishi: The Palace of the Earth Spirits serves as my humble abode.
User: Who is your sister?
Komeiji Koishi: Satori Komeiji. The one with all the serious face. 😜
```

現在、私たちの論文で議論されているテーマの範囲により、システムは単一ターンの対話のみをサポートしています。今後、複数ターンの対話をサポートするために、さらなる技術的努力を投入する予定です！
