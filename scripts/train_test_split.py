from src.preprocessing.split import create_section_split_csv


def main():

    # ===== Parameters =====
    sections_dir = r"data\sections"

    splits_dir = r"resources\splits"
    split_name = "mosfet_field_split_02"

    scenes = [
            # "JasperRidge",
            "MoffetField", 
            # "Cuprite",
          ]

    test_ratio = 0.2
    seed = 42

    # ===== Split =====
    create_section_split_csv(
        output_path=fr"{splits_dir}\{split_name}.csv",
        sections_dir=sections_dir,
        datasets=scenes,
        test_ratio=test_ratio,
        seed=seed
    )


if __name__ == "__main__":
    main()