"""
Train a K-SVD dictionary over the importance-sampled LARD patch embedding
pool produced by ``scripts/extract_patch_embeddings.py`` (paper
@sec:ksvd-setup). Writes the dictionary as an ``.npy`` file in Python
convention: shape (D, M). Also writes a ``*_val_codes.npy`` companion with
the sparse codes of a held-out validation slice for quick inspection.

Usage:
    julia --project=. scripts/julia/ksvd_lard.jl \\
        data/pretrained_seed0_ksvd_pool.h5 \\
        models/pretrained_seed0_ksvd.npy
"""

using HDF5
using KSVD
using NPZ
using LinearAlgebra
using Random
using StatsBase
using SparseArrays
using ArgParse

function parse_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "embeddings"
            help = "Path to HDF5 file with embeddings"
            required = true
        "output"
            help = "Output path for dictionary (.npy)"
            required = true
        "--dict-size", "-m"
            help = "Dictionary size (number of atoms)"
            arg_type = Int
            default = 512
        "--nnz", "-k"
            help = "Non-zeros per column (sparsity level)"
            arg_type = Int
            default = 8
        "--batch-size"
            help = "Batch size for KSVD"
            arg_type = Int
            default = 8192
        "--iters-per-batch"
            help = "KSVD iterations per batch"
            arg_type = Int
            default = 1
        "--num-repeats"
            help = "Number of passes over data"
            arg_type = Int
            default = 3
        "--val-size"
            help = "Validation set size"
            arg_type = Int
            default = 1024
    end
    return ArgParse.parse_args(s)
end

# Metrics
explainedsignal(Y, D, X; E=(Y - D * X)) = mean(norm.(eachcol(E)) ./ norm.(eachcol(Y)))
explainedvariance(Y, D, X; E=(Y - D * X)) = 1 - sum(var(E; dims=2)) / sum(var(Y; dims=2))

function main()
    args = parse_args()

    # Python writes `embeddings` with HDF5 dataspace (n, d) — shape (n, d) in
    # numpy == shape (d, n) in Julia after HDF5.jl's implicit transpose, so
    # reading gives a Julia matrix of (d, n) directly without permutedims.
    println("[KSVD] Loading embeddings from $(args["embeddings"])...")
    data = h5open(args["embeddings"], "r") do f
        read(f["embeddings"])
    end
    d, n = size(data)
    println("[KSVD] Loaded $n embeddings of dimension $d")

    # Split off validation set from the end
    val_size = min(args["val-size"], n ÷ 10)
    val_data = data[:, (end-(val_size-1)):end]
    train_end = n - val_size
    println("[KSVD] Using $val_size samples for validation, $train_end for training")

    # Config
    dict_size = args["dict-size"]
    nnz_per_col = args["nnz"]
    batch_size = min(args["batch-size"], train_end)
    iters_per_batch = args["iters-per-batch"]
    num_repeats = args["num-repeats"]

    if batch_size < args["batch-size"]
        println("[KSVD] Adjusted batch_size to $batch_size (data size: $train_end)")
    end

    sparse_coding_method = KSVD.ParallelMatchingPursuit(; max_nnz=nnz_per_col, refit_coeffs=false)

    D = nothing
    D_init = nothing

    function callback_fn((; iter, Y, D, X, norm_val, nnz_per_col_val))
        val_X = KSVD.sparse_coding(sparse_coding_method, val_data, D)
        val_var_expl = explainedvariance(val_data, D, val_X)
        val_nnz = nnz(val_X) / size(val_X, 2)
        usagecounts = countmap(X.rowval)
        numunused = length(setdiff(axes(X, 1), unique(sort(X.rowval))))
        println("  val_var_expl=$(round(val_var_expl, digits=4)), val_nnz=$(round(val_nnz, digits=1)), unused=$(numunused)")
        return nothing
    end

    println("[KSVD] Training DB-KSVD: dict_size=$dict_size, nnz=$nnz_per_col, batch=$batch_size, repeats=$num_repeats")
    for rep in 1:num_repeats
        println("[KSVD] Repeat $rep/$num_repeats")
        batch_indices = collect(Iterators.partition(1:train_end, batch_size))

        for (i, batch_idx) in enumerate(batch_indices)
            length(batch_idx) == batch_size || continue  # skip incomplete

            Y = copy(data[:, batch_idx])
            D_init = isnothing(D) ? nothing : copy(D)

            println("  Batch $i/$(length(batch_indices)), samples $(first(batch_idx))-$(last(batch_idx))")
            res = KSVD.ksvd(Y, dict_size;
                sparse_coding_method,
                verbose=false, show_trace=false,
                D_init, maxiters=iters_per_batch,
                callback_fn, abstol=nothing, reltol=nothing,
            )
            D = res.D
        end
    end

    # Save dictionary
    mkpath(dirname(args["output"]))
    println("[KSVD] Saving dictionary to $(args["output"])...")
    npzwrite(args["output"], D)

    # Save validation codes
    val_X = KSVD.sparse_coding(sparse_coding_method, val_data, D)
    val_codes_path = replace(args["output"], ".npy" => "_val_codes.npy")
    npzwrite(val_codes_path, Matrix(val_X))
    println("[KSVD] Saved validation codes to $val_codes_path")

    println("[KSVD] Done!")
end

main()
