from happytransformer import HappyTextToText, TTSettings

model_path = "./vennify/t5-base-grammar-correction"

happy_tt = HappyTextToText("T5", "model_path", use_auth_token=False)
args = TTSettings(num_beams=5, min_length=1)

result = happy_tt.generate_text("grammar: This sentences has has bads grammar.", args=args)
print(result.text) 