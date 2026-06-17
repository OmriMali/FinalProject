from src import io
from dataclasses import replace

def main():

    # ===== Source dictionary =====
<<<<<<< HEAD
    source_dir = r"results\dictionary_training\ksvd\artifacts\ksvd_test1_20260524_154408"
=======
    source_dir = r"results\dictionary_training\ksvd\artifacts\ksvd\ksvd_test1_20260617_135052"
>>>>>>> master
    source_name = "dictionary"

    # ===== Destination =====
    resources_dir = r"resources\dictionaries"
<<<<<<< HEAD
    name = "jasper_ridge_split_01_ksvd_400_atoms"
=======
    name = "cuprite_split_0_ksvd_400_atoms"
>>>>>>> master

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