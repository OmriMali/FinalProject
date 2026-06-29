from src import io
from dataclasses import replace

def main():

    # ===== Source dictionary =====
    source_dir = r"results\dictionary_training\ksvd\artifacts\ksvd\ksvd_book_results_20260629_153935"
    source_name = "dictionary"

    # ===== Destination =====
    resources_dir = r"resources\dictionaries"
    name = "jasper_ridge_split_01_project_book"

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