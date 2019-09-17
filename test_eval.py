from dlsrl.eval import f1_official_conll05

batch_bio_predicted_tags = []
batch_bio_gold_tags = []
batch_bio_predicted_tags.append(
    'B-A1 I-A1 I-A1 I-A1 B-AM-MOD O B-V B-A2 I-A2 I-A2 I-A2 B-AM-TMP I-AM-TMP O B-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV O'.split())
batch_bio_gold_tags.append(
    'B-A1 I-A1 I-A1 I-A1 B-AM-MOD O B-V B-A2 I-A2 I-A2 I-A2 B-AM-TMP I-AM-TMP O B-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV I-AM-ADV O'.split())

batch_verb_indices = [6]
batch_sentences = []
batch_sentences.append(
    "The economy 's temperature will be taken from several vantage points this week , with readings on trade , output , housing and inflation .".split())


f1_official_conll05(batch_bio_predicted_tags,   # List[List[str]]
                    batch_bio_gold_tags,        # List[List[str]]
                    batch_verb_indices,         # List[Optional[int]]
                    batch_sentences)            # List[List[str]]