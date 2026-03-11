"""Simulate text tokenization / embedding preprocessing."""
import time
import yanex

params = yanex.get_params()
corpus = params.get("corpus", "wikipedia")
vocab_size = params.get("vocab_size", 10000)
max_len = params.get("max_len", 128)

print(f"Tokenizing corpus: {corpus} (vocab={vocab_size}, max_len={max_len})")
time.sleep(0.3)

yanex.log_metrics({
    "corpus": corpus,
    "vocab_size": vocab_size,
    "max_len": max_len,
    "num_documents": 50000,
})
yanex.save_artifact({"corpus": corpus, "vocab_size": vocab_size}, "tokenizer.pkl")
print("Done.")
