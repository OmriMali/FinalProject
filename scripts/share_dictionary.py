from src import io
from dataclasses import replace

def main():

    # ===== Source dictionary =====
    source_dir = r"results\dictionary_training\ksvd\artifacts\ksvd_test1_20260524_154408"
    source_name = "dictionary"

    # ===== Destination =====
    resources_dir = r"resources\dictionaries"
    name = "jasper_ridge_split_01_ksvd_400_atoms"

    # ===== Load dictionary =====
    dictionary = io.load_dictionary(source_dir, source_name)
    replace(dictionary, name=name)

    # ===== Save to shared resources =====
    io.save_dictionary(
        dictionary,
        resources_dir,
        name,
    )

    print(
        f"Saved shared dictionary:\n"
        f"{resources_dir}\\{name}.npz"
    )


if __name__ == "__main__":
    main()