"""
Sparse-code all patch tokens against a trained K-SVD dictionary via
parallel matching pursuit. Outputs a fixed-nnz HDF5 layout (indices +
values) that Python reads as a (nnz, n_total) pair of arrays.

Expects the input HDF5 to contain a dataset called `embeddings` (paper
repo convention; see `scripts/extract_patch_embeddings.py`). Subset
metadata attrs (subset_names, subset_sizes, patches_per_image) are
copied through so downstream Python scripts don't need to hardcode them.

Usage:
    julia --project=. -t auto scripts/julia/sparse_code_patches.jl \\
        data/pretrained_seed0_all_patches.h5 \\
        models/pretrained_seed0_ksvd.npy \\
        data/pretrained_seed0_patch_codes.h5
"""

using HDF5, KSVD, NPZ, LinearAlgebra, SparseArrays, ArgParse

function parse_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "patches"
            help = "HDF5 file with all patch tokens"
            required = true
        "dictionary"
            help = "Dictionary .npy file"
            required = true
        "output"
            help = "Output .h5 file for sparse codes (fixed-nnz format)"
            required = true
        "--nnz", "-k"
            help = "Non-zeros per column"
            arg_type = Int
            default = 8
        "--chunk-size"
            help = "Columns to process at once"
            arg_type = Int
            default = 50_000
    end
    return ArgParse.parse_args(s)
end

function main()
    args = parse_args()
    nnz_k = args["nnz"]
    chunk_size = args["chunk-size"]

    println("[CODES] Loading dictionary from $(args["dictionary"])...")
    D = Float32.(npzread(args["dictionary"]))
    d, m = size(D)
    println("[CODES] Dictionary: $d × $m")

    # Pre-compute DtD once
    DtD = D' * D

    # Python wrote `embeddings` with numpy shape (n_total, d). HDF5.jl
    # reverses the dimension order to preserve the memory buffer, so the
    # Julia view is (d, n_total).
    println("[CODES] Opening patches from $(args["patches"])...")
    f_in = h5open(args["patches"], "r")
    patch_ds = f_in["embeddings"]
    d_check, n_total = size(patch_ds)
    @assert d_check == d "Dimension mismatch: dictionary $d vs patches $d_check"
    println("[CODES] $n_total patch tokens of dim $d")

    method = KSVD.ParallelMatchingPursuit(; max_nnz=nnz_k, refit_coeffs=false)

    # Create output HDF5
    # Julia dims are transposed vs Python: write (n_total, nnz_k) so Python sees (nnz_k, n_total)
    mkpath(dirname(args["output"]))
    f_out = h5open(args["output"], "w")
    ds_indices = create_dataset(f_out, "indices", Int16, (n_total, nnz_k);
                                chunk=(min(chunk_size, n_total), nnz_k))
    ds_values = create_dataset(f_out, "values", Float32, (n_total, nnz_k);
                                chunk=(min(chunk_size, n_total), nnz_k))
    f_out["num_atoms"] = m
    # Copy paper-scoped metadata from the input patches file to the codes
    # file so downstream Python (atoms.py) doesn't need hardcoded sizes.
    in_attrs = HDF5.attributes(f_in)
    out_attrs = HDF5.attributes(f_out)
    for attr_name in ("subset_names", "subset_sizes", "patches_per_image",
                      "num_images", "skipped_patches")
        if haskey(in_attrs, attr_name)
            out_attrs[attr_name] = read(in_attrs[attr_name])
        end
    end

    n_chunks = cld(n_total, chunk_size)
    println("[CODES] Processing $n_total columns in $n_chunks chunks of ≤$chunk_size...")

    for (ci, start) in enumerate(1:chunk_size:n_total)
        stop = min(start + chunk_size - 1, n_total)
        n_cols = stop - start + 1

        # Julia view is (d, n_total), so chunk samples live along dim 2.
        Y = Float32.(patch_ds[:, start:stop])

        DtY = D' * Y
        X = KSVD.sparse_coding(method, Y, D; DtD, DtY)

        # Convert sparse to fixed-nnz format
        # Write to (col_idx, r) in Julia → Python sees (r, col_idx) = (nnz_k, n_total)
        for j in 1:n_cols
            col = X[:, j]
            nz_idx = col.nzind
            nz_val = col.nzval

            for r in 1:nnz_k
                if r <= length(nz_idx)
                    ds_indices[start + j - 1, r] = Int16(nz_idx[r] - 1)  # 0-indexed for Python
                    ds_values[start + j - 1, r] = Float32(nz_val[r])
                else
                    ds_indices[start + j - 1, r] = Int16(0)
                    ds_values[start + j - 1, r] = Float32(0)
                end
            end
        end

        actual_nnz = nnz(X) / n_cols
        println("  Chunk $ci/$n_chunks (cols $start:$stop): avg nnz = $(round(actual_nnz, digits=1))")
    end

    close(f_in)
    close(f_out)
    println("[CODES] Saved fixed-nnz codes to $(args["output"])")
end

main()
