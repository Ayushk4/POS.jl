using CorpusLoaders
using MultiResolutionIterators
using TextAnalysis
using TextAnalysis: CRF, crf_loss
using WordTokenizers
using Embeddings
using Flux
using Flux: onehot, param, onehotbatch, Conv, LSTM, flip, Momentum, reset!, onecold, batchseq
using Tracker
using BSON: @save, @load
using LinearAlgebra
using Base.Iterators: partition

println("Dependencies Loaded")

device = :gpu

if device == :gpu
    using CuArrays
end

CHAR_EMBED_DIMS = 25
WORD_EMBED_DIMS = 100

train_set = load(CoNLL(), "train") # training set
# test_set = load(CoNLL(), "test") # test set
dev_set = load(CoNLL(), "dev") # dev set

dataset = flatten_levels(train_set, lvls(CoNLL, :document)) |> full_consolidate
dev_dataset = flatten_levels(dev_set, lvls(CoNLL, :document)) |> full_consolidate

get_pos(w::NERTaggedWord) = w.pos

X_train = [CorpusLoaders.word.(sent) for sent in dataset]
Y_train = [get_pos.(sent) for sent in dataset]
@assert length.(X_train) == length.(Y_train)

X_dev = [CorpusLoaders.word.(sent) for sent in dev_dataset]
Y_dev = [get_pos.(sent) for sent in dev_dataset]
@assert length.(X_dev) == length.(Y_dev)

words_vocab = unique(vcat(X_train...))
alphabets = unique(vcat(collect.(words_vocab)...))
labels = unique(vcat(Y_train...))
DENSE_OUT_SIZE = num_labels = length(labels)

#### Word Embeddings
# using these indices to generate the flux one hot input word embeddings vectors.

embtable = load_embeddings(GloVe, 2)
get_word_index = Dict(word => ii for (ii, word) in enumerate(embtable.vocab))
get_word_from_index = Dict(value => key for (key, value) in get_word_index)
W_word_Embed = (embtable.embeddings)

# One Vec for unknown chars
UNK_Word = "<UNK>"
UNK_Word_Idx = length(get_word_index) + 1
embedding_vocab_length = length(get_word_index) + 1
@assert UNK_Word ∉ collect(keys(get_word_index))
get_word_index[UNK_Word] = UNK_Word_Idx
get_word_from_index[UNK_Word_Idx] = UNK_Word
W_word_Embed = hcat(W_word_Embed, rand(WORD_EMBED_DIMS))

# One Vec for PAD chars
PAD_Word = "<pad>"
PAD_Word_Idx = length(get_word_index) + 1
embedding_vocab_length = length(get_word_index) + 1
@assert PAD_Word ∉ collect(keys(get_word_index))
get_word_index[PAD_Word] = PAD_Word_Idx
get_word_from_index[PAD_Word_Idx] = PAD_Word
W_word_Embed = hcat(W_word_Embed, rand(WORD_EMBED_DIMS))

W_word_Embed = Float32.(W_word_Embed)
@assert size(W_word_Embed, 2) == embedding_vocab_length


##### Char Embeddings
UNK_char = '¿'
PAD_char = '█'
@assert UNK_char ∉ alphabets
@assert PAD_char ∉ alphabets
push!(alphabets, UNK_char)
push!(alphabets, PAD_char)
sort!(unique!(alphabets))

# using these indices to generate the flux one hot input Char embeddings vectors.

W_Char_Embed = rand(CHAR_EMBED_DIMS, length(alphabets))
W_Char_Embed = Float32.(W_Char_Embed)
W_Char_Embed = param(W_Char_Embed)
get_char_index = Dict(char => ii for (ii, char) in enumerate(alphabets))
get_char_from_index = Dict(value => key for (key, value) in get_char_index)

println("Embeddings Made")

############# Creating onehoot vectors (useful for embeddings lookup), convenient
BATCH_SIZE = 10
X_train = sort(X_train, by=length, alg=MergeSort)
Y_train = sort(Y_train, by=length, alg=MergeSort) # For stable sorting

onehotword(word) = onehot(get(get_word_index, lowercase(word), get_word_index[UNK_Word]), 1:embedding_vocab_length)
onehotbatchword(words) = onehotbatch([get(get_word_index, lowercase(word), get_word_index[UNK_Word]) for word in words], 1:embedding_vocab_length)
onehotchars(word) = onehotbatch([get(get_char_index, c, get_char_index[UNK_char]) for c in word], 1:length(alphabets))
onehotlabel(label) = onehot(label, labels)
onehotinput(word) = (onehot(get(get_word_index, lowercase(word), get_word_index[UNK_Word]), 1:embedding_vocab_length),
                onehotbatch([get(get_char_index, c, get_char_index[UNK_char]) for c in word], 1:length(alphabets)))

# align(k) = [[j[i]  for j in k] for i in 1:length(k[1])]
# batch_char_input(ohed_chars, p) = [align(padded(b, p)) for b in collect(partition(ohed_chars, BATCH_SIZE))]
# batches(xs, p) = [batchseq(b, p) for b in partition(xs, BATCH_SIZE)]

oh_seq(arr, f) = [f(element) for element in arr]

padded(b, p, m) = [length(elem) < m ? [elem..., repeat([p], m - length(elem))...] : elem for elem in b]
padded(b, p) = padded(b, p, maximum(length.(b)))

align_chars_new(b, p, m) = [[[p, p, elem[i]..., repeat([p], m[i] - length(elem[i]))..., p, p] for i in 1:length(elem)] for elem in b]
align_chars_new(b, p) = align_chars_new(b, p, [maximum([length(j[i]) for j in b])   for i in 1:length(b[1])])

batch_oh_chars(o) = [[char_w_seq[i] for char_w_seq in o] for i in 1:length(o[1])]
gpuit(x) = cu.(x)
batch_new_chars(xs, p) = [gpuit.(batch_oh_chars(oh_seq.(align_chars_new(padded(b, p), '█'), onehotchars))) for b in partition(xs, BATCH_SIZE)]
# X_chars_train = batch_char_input([oh_seq(sentence, onehotchars) for sentence in X_train], onehotchars("█")) # A Bunch Sequences of Array of Chars, done to prevent repeated computations.
X_newchars_train = batch_new_chars([collect.(sentence) for sentence in X_train], collect("█"))

padded_word_batches(b, p, m) = [[length(elem) < i ? p : elem[i] for elem in b] for i in 1:m]
padded_word_batches(b, p) = padded_word_batches(b, p, maximum(length.(b)))
X_words_train = [cu.(oh_seq(padded_word_batches(b, PAD_Word), onehotbatchword)) for b in partition(X_train, BATCH_SIZE)] # A Bunch of Sequences of words, i.e. sentences

X_input_train = [[(k,l) for (k,l) in zip(i,j)] for (i,j) in zip(X_words_train, X_newchars_train)]
Y_oh_train = [cu.(oh_seq.(b, onehotlabel)) for b in partition(Y_train, BATCH_SIZE)]
deleteat!(Y_oh_train, length(Y_oh_train))
deleteat!(X_input_train, length(X_input_train))

X_dev = sort(X_dev, by=length, alg=MergeSort)
Y_dev = sort(Y_dev, by=length, alg=MergeSort) # For stable sorting

X_newchars_dev = batch_new_chars([collect.(sentence) for sentence in X_dev], collect("█"))
X_words_dev = [cu.(oh_seq(padded_word_batches(b, PAD_Word), onehotbatchword)) for b in partition(X_dev, BATCH_SIZE)] # A Bunch of Sequences of words, i.e. sentences
X_input_dev = [[(k,l) for (k,l) in zip(i,j)] for (i,j) in zip(X_words_dev, X_newchars_dev)]
Y_oh_dev = [oh_seq.(b, onehotlabel) for b in partition(Y_dev, BATCH_SIZE)]
# deleteat!(Y_oh_dev, length(Y_oh_dev))
# deleteat!(X_input_dev, length(X_input_dev))

# if device == :gpu
#     X_input_train = [cu.(oh_seq(sentence, onehotinput)) for sentence in X_train]
# else
#     X_input_train = [oh_seq(sentence, onehotinput) for sentence in X_train]
# end
# Y_oh_train = [oh_seq(tags_sequence, onehotlabel) for tags_sequence in Y_train]
#
# if device == :gpu
#     Y_oh_train = cu.(Y_oh_train)
# end
#
# X_input_dev = [oh_seq(sentence, onehotinput) for sentence in X_dev]
# Y_oh_dev = [oh_seq(tags_sequence, onehotlabel) for tags_sequence in Y_dev]
#
# @assert length.(X_train) == length.(X_input_train) == length.(Y_oh_train)

#################### MODEL ######################
# 1. Character Embeddings with CNNs
# 2. Obtain Word Embeddings, concatenate Char & Word Embeddings
# 3. Bi-LSTM
# 4. Softmax or CRF on each unit in sequence

CNN_OUTPUT_SIZE = 30
CONV_WINDOW_LENGTH = 3
LSTM_STATE_SIZE = 200
DROPOUT_RATE_CNN = DROPOUT_INPUT_EMBEDDING = DROPOUT_OUT_LAYER = Float32(0.5)

# 1. Dropout before conv, Max poll layer, PADDING on both sides, hyperparams = window size, output vector size.
W_Char_Embed = W_Char_Embed |> gpu
conv1 = Conv((CHAR_EMBED_DIMS, CONV_WINDOW_LENGTH), 1=>CNN_OUTPUT_SIZE, pad=(0,2)) |> gpu
dropout1 = Dropout(DROPOUT_RATE_CNN) |> gpu

slice_embed(x) = W_Char_Embed * x
char_features = Chain(x -> slice_embed.(x),
                      x -> cat(x..., dims=4),
                      dropout1,
                      conv1,
                      x -> maximum(x, dims=2),
                      x -> reshape(x, CNN_OUTPUT_SIZE, BATCH_SIZE))
# 2. Input embeddings

W_word_Embed = param(W_word_Embed) |> gpu # For trainable
get_word_embedding(w) = W_word_Embed * w |> gpu # works coz - onehot

# Dropout before LSTM
dropout_embed = Dropout(DROPOUT_INPUT_EMBEDDING) |> gpu
input_embeddings((w, cs)) = dropout_embed(vcat(get_word_embedding(w), char_features(cs))) |> gpu

# 3. Bi-LSTM

if device == :gpu
    forward_lstm = LSTM(CNN_OUTPUT_SIZE + WORD_EMBED_DIMS, LSTM_STATE_SIZE) |> gpu
    backward = LSTM(CNN_OUTPUT_SIZE + WORD_EMBED_DIMS, LSTM_STATE_SIZE) |> gpu
else
    forward_lstm = LSTM(CNN_OUTPUT_SIZE + WORD_EMBED_DIMS, LSTM_STATE_SIZE)
    backward = LSTM(CNN_OUTPUT_SIZE + WORD_EMBED_DIMS, LSTM_STATE_SIZE)
end
backward_lstm(x) = reverse(backward.(reverse(x)))

bilstm_layer(x) = vcat.(forward_lstm.(x), backward_lstm(x))

# 4. Softmax / CRF Layer
# Dropout after LSTM

if device == :gpu
    dropout2 = Dropout(DROPOUT_OUT_LAYER) |> gpu
    d_out = Dense(LSTM_STATE_SIZE * 2, DENSE_OUT_SIZE + 2) |> gpu
else
    dropout2 = Dropout(DROPOUT_OUT_LAYER)
    d_out = Dense(LSTM_STATE_SIZE * 2, DENSE_OUT_SIZE + 2)
end

m = Chain(x -> input_embeddings.(x),
                bilstm_layer,
                x -> dropout2.(x),
                x -> d_out.(x))

if device == :gpu
    m = gpu(m)
end
# dropout.(bilstm_layer(input_embeddings.(w_cs))) |> gpu

using TextAnalysis: crf_loss, CRF
Flux.@treelike TextAnalysis.CRF

c = TextAnalysis.CRF(num_labels)

init_α = Float32.(fill(-10000, (c.n + 2, 1)))
init_α[c.n + 1] = 0
if device == :gpu
    init_α = cu(init_α)
    c = c |> gpu
end

geti(x, i) = x[:, i]
loss(x, y) = sum([crf_loss(c, [geti(x[j], i) for j in 1:length(y[i])], y[i], init_α) for i in 1:BATCH_SIZE])

η = 0.01 # learning rate
β = 0.05 # rate decay
ρ = 0.9 # momentum

# TODO: rate decay
opt = Flux.Optimiser(ExpDecay(β), Momentum(η, ρ))
data = zip(X_input_train, Y_oh_train)

println("Model Built")

function try_outs()
    Flux.testmode!(dropout1)
    Flux.testmode!(dropout2)
    Flux.testmode!(dropout_embed)

    confusion_matrix = zeros(Int, (num_labels, num_labels))
    for (x, y) in zip(X_input_dev, Y_oh_dev)
        reset!(forward_lstm)
        reset!(backward)
        x_seq = m(x)
        sum([crf_loss(c, [geti(x_seq[j], i) for j in 1:length(y[i])], y[i], init_α) for i in 1:BATCH_SIZE])

        ohs = [TextAnalysis.viterbi_decode(cpu(c), cpu.([geti(x_seq[j], i) for j in 1:length(y[i])]), cpu(init_α)) for i in 1:BATCH_SIZE]

        for i in 1:BATCH_SIZE
            for d in zip(ohs[i], y[i])
                confusion_matrix[d[1].ix, d[2].ix] += 1
            end
        end
    end

    s1 = sum(confusion_matrix, dims = 2)
    s2 = sum(confusion_matrix, dims = 1)
    dg = diag(confusion_matrix)

    a = sum(dg ./ s1) / num_labels
    b = sum(dg ./ s2') / num_labels

    println("Precision and recall are:", a, " ", b)
    println("F1 is:", (2 * a * b) / (a + b))

    Flux.testmode!(dropout1, false)
    Flux.testmode!(dropout2, false)
    Flux.testmode!(dropout_embed, false)
end

function save_weights(epoch, conv1, W_word_Embed, W_Char_Embed, forward_lstm, backward, d_out, c)
    conv_cpu = conv1 |> cpu
    W_word_cpu = W_word_Embed |> cpu
    W_char_cpu = W_Char_Embed |> cpu
    forward_lstm_cpu = forward_lstm |> cpu
    backward_lstm_cpu = backward |> cpu
    crf_cpu = c |> cpu
    d_cpu = d_out |> cpu

    @save "./weights/conv_cpu$(epoch).bson" conv_cpu
    @save "./weights/W_word_cpu$(epoch).bson" W_word_cpu
    @save "./weights/W_char_cpu$(epoch).bson" W_char_cpu
    @save "./weights/forward_lstm$(epoch).bson" forward_lstm_cpu
    @save "./weights/backward_lstm$(epoch).bson" backward_lstm_cpu
    @save "./weights/d_cpu$(epoch).bson" d_cpu
    @save "./weights/crf$(epoch).bson" crf_cpu
end

UPPER_BOUND = Float32(5.0)

# Gradient Clipping
grad_clipping(g) = min(g, UPPER_BOUND)

# Backward
function back!(p::Flux.Params, l, opt)
    l = Tracker.hook(x -> grad_clipping(x), l)
    grads = Tracker.gradient(() -> l, p)
    Tracker.update!(opt, p, grads)
    return
end


function train(EPOCHS)
    reset!(forward_lstm)
    reset!(backward)

    ps = params(params(conv1)..., params(W_word_Embed)..., params(W_Char_Embed)..., params(forward_lstm, backward)..., params(d_out)..., params(c)...)
    for epoch in 1:EPOCHS
        if epoch % 2 == 1
            save_weights(epoch, conv1, W_word_Embed, W_Char_Embed, forward_lstm, backward, d_out, c)
        end
        reset!(forward_lstm)
        reset!(backward)
        try_outs()
        println("----------------------- Starting with epoch : $(epoch) ----------------------")

        for d in data
            reset!(forward_lstm)
            reset!(backward)

            l = loss(m(d[1]), d[2])
            back!(ps, l, opt)
        end
        reset!(forward_lstm)
        reset!(backward)
    end
end

train(100)
reset!(forward_lstm)
reset!(backward)
# save_weights(51, conv_cpu, W_word_Embed, W_Char_Embed, forward_lstm, backward, d_out, c)
try_outs()
