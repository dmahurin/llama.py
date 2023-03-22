import llama as llama

# Initialize llama context
params = llama.llama_context_default_params()

n = 512

params.n_ctx = n
params.n_parts = -1
params.seed = 1679473604
params.f16_kv = False
params.logits_all = False
params.vocab_only = False

# Set model path accordingly
ctx = llama.llama_init_from_file('models/ggml-model-q4.bin', params)

# Tokenize text
tokens = (llama.llama_token * n)()
n_tokens = llama.llama_tokenize(ctx, 'Q: What is the capital of France? A: ', tokens, n, True)
if n_tokens < 0:
    print('Error: llama_tokenize() returned {}'.format(n_tokens))
    exit(1)

text = "".join(llama.llama_token_to_str(ctx, t) for t in tokens[:n_tokens])
print(text)

# Evaluate tokens
for i in range(3):
    r = llama.llama_eval(ctx, tokens, n_tokens, 0, 12)
    if r != 0:
        print('Error: llama_eval() returned {}'.format(r))
        exit(1)
    token = llama.llama_sample_top_p_top_k(ctx, tokens, n_tokens , top_k=40, top_p=0.95, temp=0.8, repeat_penalty=1.1)
    print(token)
    tokens[n_tokens] = token
    n_tokens += 1
    text = "".join(llama.llama_token_to_str(ctx, t) for t in tokens[:n_tokens])
    print(text)

# # Print timings
llama.llama_print_timings(ctx)

# # Free context
llama.llama_free(ctx)